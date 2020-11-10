# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: Update kwargs with defaults

import os
import pickle
from copy import deepcopy
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmf.modules.attention import AttentionLayer, SelfAttention, SelfGuidedAttention
from mmf.modules.bottleneck import MovieBottleneck
from mmf.modules.layers import AttnPool1d, Identity
from mmf.utils.file_io import PathManager
from mmf.utils.vocab import Vocab
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.modeling_bert import BertEmbeddings


class TextEmbedding(nn.Module):
    def __init__(self, emb_type, **kwargs):
        super().__init__()
        self.model_data_dir = kwargs.get("model_data_dir", None)
        self.embedding_dim = kwargs.get("embedding_dim", None)

        # Update kwargs here
        if emb_type == "identity":
            self.module = Identity()
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "vocab":
            self.module = VocabEmbedding(**kwargs)
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "projection":
            self.module = ProjectionEmbedding(**kwargs)
            self.module.text_out_dim = self.module.out_dim
        elif emb_type == "preextracted":
            self.module = PreExtractedEmbedding(**kwargs)
        elif emb_type == "bilstm":
            self.module = BiLSTMTextEmbedding(**kwargs)
        elif emb_type == "attention":
            self.module = AttentionTextEmbedding(**kwargs)
        elif emb_type == "mcan":
            self.module = SAEmbedding(**kwargs)
        elif emb_type == "torch":
            vocab_size = kwargs["vocab_size"]
            embedding_dim = kwargs["embedding_dim"]
            self.module = nn.Embedding(vocab_size, embedding_dim)
            self.module.text_out_dim = self.embedding_dim
        else:
            raise NotImplementedError("Unknown question embedding '%s'" % emb_type)

        self.text_out_dim = self.module.text_out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class VocabEmbedding(nn.Module):
    def __init__(self, embedding_dim, **vocab_params):
        super().__init__()
        self.vocab = Vocab(**vocab_params)
        self.module = self.vocab.get_embedding(
            nn.Embedding, embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.module(x)


class BiLSTMTextEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_layers,
        dropout,
        bidirectional=False,
        rnn_type="GRU",
    ):
        super().__init__()
        self.text_out_dim = hidden_dim
        self.bidirectional = bidirectional

        if rnn_type == "LSTM":
            rnn_cls = nn.LSTM
        elif rnn_type == "GRU":
            rnn_cls = nn.GRU

        self.recurrent_encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.recurrent_encoder(x)
        # Return last state
        if self.bidirectional:
            return out[:, -1]

        forward_ = out[:, -1, : self.num_hid]
        backward = out[:, 0, self.num_hid :]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        output, _ = self.recurrent_encoder(x)
        return output


class PreExtractedEmbedding(nn.Module):
    def __init__(self, out_dim, base_path):
        super().__init__()
        self.text_out_dim = out_dim
        self.base_path = base_path
        self.cache = {}

    def forward(self, qids):
        embeddings = []
        for qid in qids:
            embeddings.append(self.get_item(qid))
        return torch.stack(embeddings, dim=0)

    @lru_cache(maxsize=5000)
    def get_item(self, qid):
        return np.load(os.path.join(self.base_path, str(qid.item()) + ".npy"))


class AttentionTextEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super().__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]

        bidirectional = kwargs.get("bidirectional", False)

        self.recurrent_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        self.recurrent_unit.flatten_parameters()
        # self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(x)  # N * T * hidden_dim
        lstm_drop = self.dropout(lstm_out)  # N * T * hidden_dim
        lstm_reshape = lstm_drop.permute(0, 2, 1)  # N * hidden_dim * T

        qatt_conv1 = self.conv1(lstm_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        return qtt_feature_concat


class ProjectionEmbedding(nn.Module):
    def __init__(self, module, in_dim, out_dim, **kwargs):
        super().__init__()
        if module == "linear":
            self.layers = nn.Linear(in_dim, out_dim)
            self.out_dim = out_dim
        elif module == "conv":
            last_out_channels = in_dim
            layers = []
            for conv in kwargs["convs"]:
                layers.append(nn.Conv1d(in_channels=last_out_channels, **conv))
                last_out_channels = conv["out_channels"]
            self.layers = nn.ModuleList(*layers)
            self.out_dim = last_out_channels
        else:
            raise TypeError(
                "Unknown module type for 'ProjectionEmbedding',"
                "use either 'linear' or 'conv'"
            )

    def forward(self, x):
        return self.layers(x)


class ImageFeatureEmbedding(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    """

    def __init__(self, img_dim, question_dim, **kwargs):
        super().__init__()

        self.image_attention_model = AttentionLayer(img_dim, question_dim, **kwargs)
        self.out_dim = self.image_attention_model.out_dim

    def forward(self, image_feat_variable, question_embedding, image_dims, extra=None):
        if extra is None:
            extra = {}
        # N x K x n_att
        attention = self.image_attention_model(
            image_feat_variable, question_embedding, image_dims
        )
        att_reshape = attention.permute(0, 2, 1)

        order_vectors = getattr(extra, "order_vectors", None)

        if order_vectors is not None:
            image_feat_variable = torch.cat(
                [image_feat_variable, order_vectors], dim=-1
            )
        tmp_embedding = torch.bmm(
            att_reshape, image_feat_variable
        )  # N x n_att x image_dim
        batch_size = att_reshape.size(0)
        image_embedding = tmp_embedding.view(batch_size, -1)

        return image_embedding, attention


class MultiHeadImageFeatureEmbedding(nn.Module):
    def __init__(self, img_dim, question_dim, **kwargs):
        super().__init__()
        self.module = nn.MultiheadAttention(
            embed_dim=question_dim, kdim=img_dim, vdim=img_dim, **kwargs
        )
        self.out_dim = question_dim

    def forward(self, image_feat_variable, question_embedding, image_dims, extra=None):
        if extra is None:
            extra = {}
        image_feat_variable = image_feat_variable.transpose(0, 1)
        question_embedding = question_embedding.unsqueeze(1).transpose(0, 1)
        output, weights = self.module(
            question_embedding, image_feat_variable, image_feat_variable
        )
        output = output.transpose(0, 1)

        return output.squeeze(), weights


class ImageFinetune(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super().__init__()
        with PathManager.open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with PathManager.open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3


class BertVisioLinguisticEmbeddings(BertEmbeddings):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.token_type_embeddings_visual = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings_visual = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def initialize_visual_from_pretrained(self):
        self.token_type_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.token_type_embeddings.weight.data), requires_grad=True
        )
        self.position_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.position_embeddings.weight.data), requires_grad=True
        )

    def encode_text(
        self, input_ids: Tensor, token_type_ids: Optional[Tensor] = None
    ) -> Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        return embeddings

    def encode_image(
        self,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        image_text_alignment: Optional[Tensor] = None,
    ) -> Tensor:

        visual_embeddings = self.projection(visual_embeddings)
        token_type_embeddings_visual = self.token_type_embeddings_visual(
            visual_embeddings_type
        )

        # get position_embeddings
        # this depends on image_text_alignment
        position_embeddings_visual = self.get_position_embeddings_visual(
            visual_embeddings, image_text_alignment=image_text_alignment
        )

        # calculate visual embeddings
        v_embeddings = (
            visual_embeddings
            + position_embeddings_visual
            + token_type_embeddings_visual
        )
        return v_embeddings

    def get_position_embeddings_visual(
        self, visual_embeddings: Tensor, image_text_alignment: Optional[Tensor] = None
    ) -> Tensor:

        if image_text_alignment is not None:
            # image_text_alignment = Batch x image_length x alignment_number.
            # Each element denotes the position of the word corresponding to the
            # image feature. -1 is the padding value.
            image_text_alignment_mask = (
                (image_text_alignment != -1).long().to(image_text_alignment.device)
            )
            # Get rid of the -1.
            image_text_alignment = image_text_alignment_mask * image_text_alignment

            # position_embeddings_visual
            # = Batch x image_length x alignment length x dim
            position_embeddings_visual = self.position_embeddings(
                image_text_alignment
            ) * image_text_alignment_mask.unsqueeze(-1)
            position_embeddings_visual = position_embeddings_visual.sum(2)

            # We want to averge along the alignment_number dimension.
            image_text_alignment_mask = image_text_alignment_mask.sum(2)
            image_text_alignment_mask[image_text_alignment_mask == 0] = torch.tensor(
                [1], dtype=torch.long
            )  # Avoid devide by zero error
            position_embeddings_visual = (
                position_embeddings_visual / image_text_alignment_mask.unsqueeze(-1)
            )

            position_ids_visual = torch.zeros(
                visual_embeddings.size()[:-1],
                dtype=torch.long,
                device=visual_embeddings.device,
            )

            position_embeddings_visual = (
                position_embeddings_visual
                + self.position_embeddings_visual(position_ids_visual)
            )
        else:
            position_ids_visual = torch.zeros(
                visual_embeddings.size()[:-1],
                dtype=torch.long,
                device=visual_embeddings.device,
            )
            position_embeddings_visual = self.position_embeddings_visual(
                position_ids_visual
            )

        return position_embeddings_visual

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        image_text_alignment: Optional[Tensor] = None,
    ) -> Tensor:
        """
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embedding = [batch_size, image_feature_length, image_feature_dim]
        image_text_alignment = [batch_size, image_feature_length, alignment_dim]
        """

        # text embeddings
        text_embeddings = self.encode_text(input_ids, token_type_ids=token_type_ids)

        # visual embeddings
        if visual_embeddings is not None and visual_embeddings_type is not None:
            v_embeddings = self.encode_image(
                visual_embeddings,
                visual_embeddings_type=visual_embeddings_type,
                image_text_alignment=image_text_alignment,
            )

            # Concate the two:
            embeddings = torch.cat(
                (text_embeddings, v_embeddings), dim=1
            )  # concat the visual embeddings after the attentions

        else:
            embeddings = text_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SAEmbedding(nn.Module):
    """Encoder block implementation in MCAN https://arxiv.org/abs/1906.10770
    """

    def __init__(self, hidden_dim: int, embedding_dim: int, **kwargs):
        super().__init__()
        num_attn = kwargs["num_attn"]
        num_layers = kwargs["num_layers"]
        dropout = kwargs.get("dropout", 0.1)
        num_attn_pool = kwargs.get("num_attn_pool", 1)
        num_feat = kwargs.get("num_feat", -1)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.self_attns = nn.ModuleList(
            [SelfAttention(hidden_dim, num_attn, dropout) for _ in range(num_layers)]
        )
        self.attn_pool = None
        self.num_feat = num_feat
        self.text_out_dim = hidden_dim
        if num_attn_pool > 0:
            self.attn_pool = AttnPool1d(hidden_dim, num_feat * num_attn_pool)
            self.text_out_dim = hidden_dim * num_attn_pool

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        out, (h, c) = self.lstm(x)
        for self_attn in self.self_attns:
            out = self_attn(out, mask)

        vec = h.transpose(0, 1).contiguous().view(b, 1, -1)
        if self.attn_pool:
            vec = self.attn_pool(out, out, mask).view(b, self.num_feat, -1)

        return out, vec


class SGAEmbedding(nn.Module):
    """Decoder block implementation in MCAN https://arxiv.org/abs/1906.10770
    """

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        num_attn = kwargs["num_attn"]
        num_layers = kwargs["num_layers"]
        dropout = kwargs.get("dropout", 0.1)
        hidden_dim = kwargs.get("hidden_dim", 512)

        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.self_guided_attns = nn.ModuleList(
            [
                SelfGuidedAttention(hidden_dim, num_attn, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.view(b, c, -1).transpose(1, 2).contiguous()  # b x (h*w) x c

        x = self.linear(x)

        for self_guided_attn in self.self_guided_attns:
            x = self_guided_attn(x, y, x_mask, y_mask)

        return x


class CBNEmbedding(nn.Module):
    """MoVie bottleneck layers from https://arxiv.org/abs/2004.11883
    """

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        cond_dim = kwargs["cond_dim"]
        num_layers = kwargs["cbn_num_layers"]
        compressed = kwargs.get("compressed", True)
        use_se = kwargs.get("use_se", True)

        self.out_dim = 1024
        self.layer_norm = nn.LayerNorm(self.out_dim)
        cbns = []
        for i in range(num_layers):
            if embedding_dim != self.out_dim:
                downsample = nn.Conv2d(
                    embedding_dim, self.out_dim, kernel_size=1, stride=1, bias=False
                )
                cbns.append(
                    MovieBottleneck(
                        embedding_dim,
                        self.out_dim // 4,
                        cond_dim,
                        downsample=downsample,
                        compressed=compressed,
                        use_se=use_se,
                    )
                )
            else:
                cbns.append(
                    MovieBottleneck(
                        embedding_dim,
                        self.out_dim // 4,
                        cond_dim,
                        compressed=compressed,
                        use_se=use_se,
                    )
                )
            embedding_dim = self.out_dim
        self.cbns = nn.ModuleList(cbns)
        self._init_layers()

    def _init_layers(self) -> None:
        for cbn in self.cbns:
            cbn.init_layers()

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        for cbn in self.cbns:
            x, _ = cbn(x, v)

        x = self.layer_norm(
            nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(3).squeeze(2)
        )

        return x


class TwoBranchEmbedding(nn.Module):
    """Attach MoVie into MCAN model as a counting module in
    https://arxiv.org/abs/2004.11883
    """

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        hidden_dim = kwargs.get("hidden_dim", 512)
        self.sga = SGAEmbedding(embedding_dim, **kwargs)
        self.sga_pool = AttnPool1d(hidden_dim, 1)
        self.cbn = CBNEmbedding(embedding_dim, **kwargs)
        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_sga = self.sga(x, y, x_mask, y_mask)
        x_sga = self.sga_pool(x_sga, x_sga, x_mask).squeeze(1)
        x_cbn = self.cbn(x, v)

        return x_sga, x_cbn


############## Text2Image Encoder-Decoder #######
class RNNEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(RNNEmbedding, self).__init__()
        self.ntoken = kwargs["vocab_size"]
        self.ninput = kwargs["embed_size"]  # size of each embedding vector
        self.drop_prob = kwargs["drop_prob"]  # probability of an element to be zeroed
        self.nlayers = kwargs["nlayers"]  # Number of recurrent layers
        self.bidirectional = kwargs["bidirectional"]
        self.rnn_type = kwargs["rnn_type"]
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = kwargs["nhidden"] // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            # TODO: check nlayers
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers * self.num_directions, bsz, self.nhidden),
                    weight.new_zeros(self.nlayers * self.num_directions, bsz, self.nhidden))
        else:
            return weight.new_zeros(self.nlayers * self.num_directions, bsz, self.nhidden)

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        emb = pack_padded_sequence(emb, cap_lens.cpu(), batch_first=True, enforce_sorted=False)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        # TODO: Check transpose!
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class InceptionEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(InceptionEmbedding, self).__init__()
        self.nef = kwargs["nef"]

        inception_model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
        self.define_module(inception_model)
        self.init_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.emb_features = nn.Conv2d(768, self.nef, kernel_size=1, stride=1, padding=0, bias=False)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        # Image region features
        features = self.emb_features(x)

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)

        # Global image features
        cnn_code = self.emb_cnn_code(x)

        return features, cnn_code
