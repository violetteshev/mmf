# Copyright (c) Facebook, Inc. and its affiliates.
import copy

import omegaconf
import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.embeddings import RNNEmbedding
from mmf.modules.gan_models import DMGAN_G
from torch import nn


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

    def _build_generator(self):
        generator_config = self.config["generator"][0]
        if generator_config["type"] == "DMGAN":
            self.generator = DMGAN_G(**generator_config["params"])

        # Initialize weights
        self.generator.apply(self._weights_init)

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

    def _weights_init(m):
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

    def forward(self, sample_list):
        t = sample_list
        model_output = {"scores": 0}

        return model_output
