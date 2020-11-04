# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import warnings

from mmf.common.registry import registry
from mmf.datasets.builders.vqa2.builder import VQA2Builder
from mmf.datasets.builders.vqa2.gan_vqa2_dataset import GANVQA2Dataset


@registry.register_builder("gan_vqa2")
class GANVQA2Builder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "gan_vqa2"
        self.dataset_class = GANVQA2Dataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/vqa2/gan.yaml"
