# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import os
import pickle

from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.databases.annotation_database import AnnotationDatabase
from mmf.datasets.databases.features_database import FeaturesDatabase
from mmf.datasets.databases.image_database import ImageDatabase


class MMFDataset(BaseDataset):
    """This dataset is useful for external open source dataset which
    usually have annotation files, images and features (which we generate).
    The dataset takes care of creating annotation db, features db and image db
    if the configuration follows a set format. Also, you can optionally enable
    image or features. The class has a resources method which can be overridden
    to download data. More details to come.
    """

    def __init__(
        self, dataset_name, config, dataset_type="train", index=0, *args, **kwargs
    ):
        super().__init__(dataset_name, config, dataset_type, *args, **kwargs)
        self._index = index
        self.annotation_db = self.build_annotation_db()

        self._use_images = self.config.get("use_images", False)
        if self._use_images:
            self.image_db = self.build_image_db()

        self._use_features = self.config.get("use_features", False)
        if self._use_features:
            self.features_db = self.build_features_db()
        
        self._use_ontology = self.config.get("use_ontology", False)
        if self._use_ontology:
            self.ontology = self.build_ontology()
            self.max_entity_len = self.config.get("max_entity_len", 3)

    def build_annotation_db(self):
        annotation_path = self._get_path_based_on_index(
            self.config, "annotations", self._index
        )
        return AnnotationDatabase(self.config, annotation_path)

    def build_features_db(self):
        features_path = self._get_path_based_on_index(
            self.config, "features", self._index
        )
        return FeaturesDatabase(
            self.config, features_path, annotation_db=self.annotation_db
        )

    def build_image_db(self):
        image_path = self._get_path_based_on_index(self.config, "images", self._index)
        return ImageDatabase(self.config, image_path, annotation_db=self.annotation_db)

    def build_ontology(self):
        ontology_path = self._get_path_based_on_index(
            self.config, "ontology", self._index
        )
        with open(ontology_path, 'rb') as f:
            ontology = pickle.load(f)
        return ontology

    def _get_path_based_on_index(self, config, attribute, index):
        if attribute not in config:
            raise ValueError(f"{attribute} not present in config")

        config = config.get(attribute, None)

        if (
            self.dataset_type not in config
            or len(config.get(self.dataset_type, [])) == 0
        ):
            raise ValueError(f"No {attribute} present for type {self.dataset_type}")

        paths = config[self.dataset_type]

        if isinstance(paths, str):
            selected_path = paths
        else:
            assert isinstance(paths, collections.abc.MutableSequence)
            selected_path = paths[self._index]

        selected_path = self._add_root_dir(selected_path)

        return selected_path

    def _add_root_dir(self, path):
        path = path.split(",")
        for idx, p in enumerate(path):
            path[idx] = os.path.join(self.config.data_dir, p)

        return ",".join(path)

    def __len__(self):
        return len(self.annotation_db)
