# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, Dict

import torch
from torch import Tensor, nn
from transformers.modeling_bert import BertPooler, BertPredictionHeadTransform

from mmf.common.registry import registry
from mmf.common.typings import DictConfig
from mmf.models.transformers.base import BaseTransformer, BaseTransformerInput
from mmf.modules.encoders import MultiModalEncoderBase


class ImageEncoder(MultiModalEncoderBase):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        self.encoder = self._build_modal_encoder(self.config.image_encoder)

    def forward(self, x):
        return self.encoder(x)


class MMFTransformerEmbeddings(nn.Module):
    def __init__(self, config, transformer, img_dim, img_pos_dim):
        super().__init__()
        # Text Embeddings
        # self.word_embeddings = nn.Embedding(
        #     config.vocab_size, config.hidden_size, padding_idx=0
        # )
        # self.position_embeddings = nn.Embedding(
        #     config.max_position_embeddings, config.hidden_size
        # )
        self.word_embeddings = transformer.embeddings.word_embeddings
        self.position_embeddings = transformer.embeddings.position_embeddings
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        # self.layer_norm = transformer.embeddings.LayerNorm
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Image Embeddings
        self.img_embeddings = nn.Sequential(
            nn.Linear(img_dim, config.hidden_size),
            torch.nn.LayerNorm(config.hidden_size, eps=1e-12),
        )
        self.img_pos_embeddings = nn.Sequential(
            nn.Linear(img_pos_dim, config.hidden_size),
            # torch.nn.LayerNorm(config.hidden_size, eps=1e-12),
        )
        self.img_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.img_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Token Type Embeddings
        # self.token_type_embeddings = nn.Embedding(
        #     config.type_vocab_size, config.hidden_size
        # )
        self.token_type_embeddings = transformer.embeddings.token_type_embeddings

    def forward(
        self,
        input_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
    ):
        ## Calculate text embeddings for word, position, token type
        words_embeddings = self.word_embeddings(input_ids["text"])
        # Add position ids for text tokens
        if "text" not in position_ids:
            position_ids["text"] = input_ids["text"].new_tensor(
                torch.arange(0, input_ids["text"].size(1), dtype=torch.long)
                .unsqueeze(0)
                .expand(input_ids["text"].size(0), input_ids["text"].size(1))
            )
        position_embeddings = self.position_embeddings(position_ids["text"])
        if "text" not in segment_ids:
            segment_ids["text"] = torch.zeros_like(input_ids["text"])
        txt_type_embeddings = self.token_type_embeddings(segment_ids["text"])

        txt_embeddings = self.layer_norm(
            words_embeddings + position_embeddings + txt_type_embeddings
        )
        txt_embeddings = self.dropout(txt_embeddings)

        ## Calculate image embeddings for feature, position, token type
        transformed_input = self.img_embeddings(input_ids["image"])
        img_embeddings = transformed_input
        if "image" in position_ids:
            # transformed_pos = self.img_pos_embeddings(position_ids["image"])
            transformed_pos = self.position_embeddings(position_ids["image"])
            img_embeddings += transformed_pos

        if "image" not in segment_ids:
            segment_ids["image"] = torch.zeros_like(
                input_ids["image"][:, :, 0], dtype=torch.long
            )
        img_type_embeddings = self.token_type_embeddings(segment_ids["image"])

        img_embeddings += img_type_embeddings
        img_embeddings = self.img_dropout(self.img_layer_norm(img_embeddings))

        return torch.cat([txt_embeddings, img_embeddings], dim=1)


@registry.register_model("mmf_transformer")
class MMFTransformer(BaseTransformer):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/mmf_transformer/defaults.yaml"

    def build_encoders(self):
        self.image_encoder = ImageEncoder(self.config)

    def build_embeddings(self):
        self.embeddings = MMFTransformerEmbeddings(
            self.transformer_config,
            self.transformer,
            self.config.visual_embedding_dim,
            self.config.visual_position_dim,
        )

    def init_weights(self):
        self.classifier.apply(self._init_weights)

    def build_heads(self):
        self.classifier = nn.Sequential(
            BertPooler(self.transformer_config),
            nn.Dropout(self.transformer_config.hidden_dropout_prob),
            BertPredictionHeadTransform(self.transformer_config),
            nn.Linear(self.transformer_config.hidden_size, self.config.num_labels),
        )

    def preprocess_sample(self, sample_list: Dict[str, Any]) -> BaseTransformerInput:
        # Input IDs
        input_ids: Dict[str, Tensor] = {}
        input_ids["text"] = sample_list.input_ids
        if "image_feature_0" in sample_list:
            input_ids["image"] = sample_list.image_feature_0
        elif "image" in sample_list:
            input_ids["image"] = self.image_encoder(sample_list.image)

        # Position IDs
        position_ids: Dict[str, Tensor] = {}
        position_ids["image"] = input_ids["image"].new_tensor(
            torch.arange(0, input_ids["image"].size(1), dtype=torch.long)
            .unsqueeze(0)
            .expand(input_ids["image"].size(0), input_ids["image"].size(1)),
            dtype=torch.long,
        )

        # Segment IDs
        segment_ids: Dict[str, Tensor] = {}
        segment_ids["text"] = sample_list.segment_ids

        # Masks
        masks: Dict[str, Tensor] = {}
        masks["text"] = sample_list.input_mask
        if "image_mask" in sample_list:
            masks["image"] = sample_list.image_mask
        else:
            masks["image"] = torch.ones_like(
                input_ids["image"][:, :, 0], dtype=torch.long
            )

        return BaseTransformerInput(input_ids, position_ids, segment_ids, masks)

    def forward(self, sample_list: Dict[str, Any]) -> Dict[str, Tensor]:
        # Sample preprocess
        output = self.preprocess_sample(sample_list)

        # Transformer Input Embeddings
        embedding_output = self.embeddings(
            input_ids=output.input_ids,
            position_ids=output.position_ids,
            segment_ids=output.segment_ids,
        )

        # Transformer Attention mask
        attention_mask = torch.cat(
            (output.masks["text"], output.masks["image"]), dim=-1
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Transformer Encoder
        encoded_layers = self.transformer.encoder(
            embedding_output,  # combined embedding
            extended_attention_mask,  # combined attention mask
            [None] * len(self.transformer.encoder.layer),  # head masks
        )

        # Transformer Heads
        head_output = self.classifier(encoded_layers[0])

        # Calculate losses, return postprocess outputs
        return self.postprocess_output(head_output)

    def postprocess_output(self, output: Tensor) -> Dict[str, Tensor]:
        output_dict = {}
        output_dict["scores"] = output.contiguous().view(-1, self.config.num_labels)
        return output_dict
