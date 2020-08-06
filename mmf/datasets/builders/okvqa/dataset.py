# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Type, Union

import torch

from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.mmf_dataset import MMFDataset


class OKVQADataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "okvqa"
        super().__init__(name, config, dataset_type, index, *args, **kwargs)
        self._should_fast_read = self.config.get("fast_read", False)

    def try_fast_read(self):
        # Don't fast read in case of test set.
        if self._dataset_type == "test":
            return

        if hasattr(self, "_should_fast_read") and self._should_fast_read is True:
            self.writer.write(
                "Starting to fast read {} {} dataset".format(
                    self.dataset_name, self.dataset_type
                )
            )
            self.cache = {}
            for idx in range(len(self.annotation_db)):
                self.cache[idx] = self.load_item(idx)
            self.writer.write("Finish fast read")

    def __getitem__(self, idx):
        if self._should_fast_read is True and self._dataset_type != "test":
            return self.cache[idx]
        else:
            return self.load_item(idx)
    
    def load_item(self, idx: int) -> Type[Sample]:
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

        if self._use_features is True:
            features = self.features_db[idx]
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )
            current_sample.update(features)

        if "answers" in sample_info:
            answers = self.answer_processor({"answers": sample_info["answers"]})
            current_sample.targets = answers["answers_scores"]

        return current_sample

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
