# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import torch
import tqdm
import numpy as np
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import is_master


logger = logging.getLogger(__name__)


class OKVQADataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "okvqa"
        super().__init__(name, config, dataset_type, index=imdb_file_index)
        self._should_fast_read = self.config.get("fast_read", False)

    def init_processors(self):
        super().init_processors()
        if not self._use_features:
            self.image_db.transform = self.image_processor
    
    def try_fast_read(self):
        # Don't fast read in case of test set.
        if self._dataset_type == "test":
            return

        if hasattr(self, "_should_fast_read") and self._should_fast_read:
            logger.info(
                f"Starting to fast read {self.dataset_name} {self.dataset_type} "
                + "dataset"
            )
            self.cache = {}
            # for idx in tqdm.tqdm(
            #     range(len(self.annotation_db)), miniters=100, disable=not is_master()
            # ):
            for idx in range(len(self.annotation_db)):
                self.cache[idx] = self.load_item(idx)

    def __getitem__(self, idx):
        if self._should_fast_read and self._dataset_type != "test":
            return self.cache[idx]
        else:
            return self.load_item(idx)
    
    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        if "question_tokens" in sample_info:
            text_processor_argument = {
                "tokens": sample_info["question_tokens"],
                "text": sample_info["question_str"],
            }
        else:
            text_processor_argument = {"text": sample_info["question"]}

        processed_question = self.text_processor(text_processor_argument)

        current_sample.text = processed_question["text"]
        if "input_ids" in processed_question:
            current_sample.update(processed_question)

        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        current_sample.text_len = torch.tensor(
            len(sample_info["question_tokens"]), dtype=torch.int
        )

        if self._use_features:
            features = self.features_db[idx]
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )
            current_sample.update(features)
        else:
            image_path = sample_info["image_name"] + ".jpg"
            current_sample.image = self.image_db.from_path(image_path)["images"][0]

        # Add details for knowledge entities
        current_sample = self.add_entity_details(current_sample)
        # Depending on whether we are using soft copy this can add
        # dynamic answer space
        current_sample = self.add_answer_info(sample_info, current_sample)

        return current_sample

    def add_entity_details(self, sample):
        if self._use_ontology:
            # Extract entities
            entity_ids, entity_embeds, words = self.ontology.extract_entities(sample.text)
            
            # Pad with -1 
            max_seq_len = len(sample.input_ids)
            entity_ids = [ids + [-1] * (self.ontology.max_len - len(ids)) for ids in entity_ids]

            while len(entity_embeds) < max_seq_len:
                entity_embeds.append(np.zeros(self.ontology.embed_dim))
                entity_ids.append([-1] * self.ontology.max_len)

            # Convert to tensors
            sample.entity_ids = torch.tensor(entity_ids, dtype=torch.int)
            sample.entity_embeds = torch.tensor(entity_embeds, dtype=torch.float)

        return sample

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}

            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)
            sample.targets = processed_soft_copy_answers["answers_scores"]

        return sample
    
    def idx_to_answer(self, idx):
        return self.answer_processor.convert_idx_to_answer(idx)

    def format_for_prediction(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
                if answer == self.context_processor.PAD_TOKEN:
                    answer = "unanswerable"
            else:
                answer = self.answer_processor.idx2word(answer_id)
            # actual_answer = report.answers[idx]

            predictions.append(
                {
                    "question_id": question_id.item(),
                    "answer": answer,
                    # "actual_answers": actual_answer,
                    # "question_tokens": report.question_tokens[idx],
                    # "image_id": report.image_id[idx].item()
                }
            )

        return predictions
