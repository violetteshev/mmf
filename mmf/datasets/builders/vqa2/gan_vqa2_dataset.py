# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import torch
from mmf.common.sample import Sample


logger = logging.getLogger(__name__)

from mmf.datasets.builders.vqa2.dataset import VQA2Dataset


class GANVQA2Dataset(VQA2Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = "gan_vqa2"
        self.max_pred_ans = args[0]["max_pred_ans"]

    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        # Take only max_pred_ans from predicted answers
        pred_answers = sample_info["pred_answers"][:self.max_pred_ans]
        quest_text = sample_info["question_str"]
        
        # Create targets, 1 for correct predicted answers, -1 for incorrect 
        targets = (-1)*torch.ones(len(pred_answers))
        correct_idx = [idx for idx, ans in enumerate(pred_answers) if ans in sample_info["all_answers"]]
        targets[correct_idx] = 1
        current_sample.targets = targets

        # Append each answer to question and peocess
        all_text = []
        all_length = []
        for ans in pred_answers:
            text_processor_argument = {"text": " ".join([quest_text, ans])}
            processed_question = self.text_processor(text_processor_argument)
            all_text.append(processed_question["text"])
            all_length.append(processed_question["length"])

        current_sample.captions = torch.stack(all_text, dim=0)
        current_sample.cap_len = torch.stack(all_length, dim=0)

        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

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

        return current_sample

