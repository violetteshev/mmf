# Copyright (c) Facebook, Inc. and its affiliates.
import copy

import omegaconf
import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.embeddings import (
    ImageFeatureEmbedding,
    PreExtractedEmbedding,
    TextEmbedding,
)
from mmf.modules.layers import ClassifierLayer, ModalCombineLayer
from mmf.utils.build import build_image_encoder
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
        self.model = nn.Linear(300, 200)

    def forward(self, sample_list):
        t = sample_list
        model_output = {"scores": 0}

        return model_output
