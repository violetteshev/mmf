# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.okvqa.dataset import OKVQADataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("okvqa")
class OKVQABuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="okvqa", dataset_class=OKVQADataset, *args, **kwargs):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)
        self.dataset_class = OKVQADataset


    @classmethod
    def config_path(cls):
        return "configs/datasets/okvqa/defaults.yaml"

    # def load(self, *args, **kwargs):
    #    dataset = super().load(*args, **kwargs)
    #    if dataset is not None and hasattr(dataset, "try_fast_read"):
    #        dataset.try_fast_read()
    
    #    return dataset