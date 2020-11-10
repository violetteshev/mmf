# Copyright (c) Facebook, Inc. and its affiliates.
import copy
from random import sample

import omegaconf
import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.embeddings import RNNEmbedding, InceptionEmbedding
from mmf.modules.gan_models import DMGAN_G
from torch import nn


def weights_init(m):
    # orthogonal_
    # xavier_uniform_(
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #print(m.state_dict().keys())
        if list(m.state_dict().keys())[0] == 'weight':
            nn.init.orthogonal_(m.weight.data, 1.0)
        elif list(m.state_dict().keys())[3] == 'weight_bar':
            nn.init.orthogonal_(m.weight_bar.data, 1.0)
        #nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


@registry.register_model("vqagan")
class VQAGAN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/vqagan/defaults.yaml"

    def build(self):
        self._build_text_encoder()
        self._build_image_encoder()
        self._build_generator()

    def _build_text_encoder(self):
        text_embedding_config = self.config["text_embeddings"][0]
        if text_embedding_config["type"] == "rnn":
            self.text_encoder = RNNEmbedding(**text_embedding_config["params"])
        
        # Load pretrained encoder
        encoder_path = self.config["text_encoder_path"]
        if encoder_path != '':
            state_dict = torch.load(encoder_path, map_location=lambda storage, loc: storage)
            self.text_encoder.load_state_dict(state_dict)
        
        # Freeze text encoder
        if not self.config["train_text_encoder"]:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()

    def _build_image_encoder(self):
        image_embedding_config = self.config["image_embeddings"][0]
        if image_embedding_config["type"] == "inception":
            self.image_encoder = InceptionEmbedding(**image_embedding_config["params"])
        
        # Load pretrained encoder
        encoder_path = self.config["image_encoder_path"]
        if encoder_path != '':
            state_dict = torch.load(encoder_path, map_location=lambda storage, loc: storage)
            self.image_encoder.load_state_dict(state_dict)
        
        # Freeze image encoder
        if not self.config["train_image_encoder"]:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            self.image_encoder.eval()

    def _build_generator(self):
        generator_config = self.config["generator"][0]
        self.z_dim = generator_config["params"]["z_dim"]
        
        if generator_config["type"] == "DMGAN":
            self.generator = DMGAN_G(**generator_config["params"])

        # Initialize weights
        self.generator.apply(weights_init)

        # Load pretrained generator
        generator_path = self.config["generator_path"]
        if generator_path != '':
            state_dict = torch.load(generator_path, map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(state_dict)
        
        # Freeze generator
        if not self.config["train_generator"]:
            for p in self.generator.parameters():
                p.requires_grad = False
            self.generator.eval()

    def get_optimizer_parameters(self, config):
        params = []
        if self.config["train_generator"]:
            params.append({"params": self.generator.parameters()})
        
        if self.config["train_text_encoder"]:
            params.append({"params": self.text_encoder.parameters()})

        return params

    def forward(self, sample_list):
        captions = sample_list["captions"]
        cap_lens = sample_list["cap_len"]
        real_imgs = sample_list["image"]

        #real_batch_size = captions.size(0)
        #caption_num = captions.size(1)
       
        # View captions as one batch
        batch_size = captions.size(0)*captions.size(1)
        captions = captions.view(batch_size, -1)
        cap_lens = cap_lens.view(batch_size)

        # Get text embeddings
        hidden = self.text_encoder.init_hidden(batch_size) # TODO: should it require grad?
        words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]

        # Generate images
        noise = torch.randn(batch_size, self.z_dim, device=words_embs.device)
        #noise = torch.randn(real_batch_size, 1, self.z_dim, device=words_embs.device)
        #noise = noise.repeat(1, caption_num, 1).view(batch_size, -1)

        fake_imgs, _, mu, logvar = self.generator(noise, sent_emb, words_embs, mask, cap_lens)

        # Extract image features
        fake_region_features, fake_cnn_code = self.image_encoder(fake_imgs[-1])
        real_region_features, real_cnn_code = self.image_encoder(real_imgs)

        model_output = {
            "image": fake_imgs[-1],
            "region_features": fake_region_features,
            "cnn_code": fake_cnn_code,
            "gt_region_features": real_region_features,
            "gt_cnn_code": real_cnn_code,     
        }

        return model_output
