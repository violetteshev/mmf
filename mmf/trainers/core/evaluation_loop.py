# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from abc import ABC
from random import randint
from typing import Any, Dict, Tuple, Type

import torch
import tqdm
import numpy as np
from PIL import Image
from mmf.common.meter import Meter
from mmf.common.report import Report
from mmf.common.sample import to_device
from mmf.utils.distributed import is_master


logger = logging.getLogger(__name__)


class TrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, loader, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_master()
            combined_report = None

            for batch in tqdm.tqdm(loader, disable=disable_tqdm):
                report = self._forward(batch)
                self.update_meter(report, meter)

                # accumulate necessary params for metric calculation
                if combined_report is None:
                    combined_report = report
                else:
                    combined_report.accumulate_tensor_fields(
                        report, self.metrics.required_params
                    )
                    combined_report.batch_size += report.batch_size

                if single_batch is True:
                    break

            combined_report.metrics = self.metrics(combined_report, combined_report)
            self.update_meter(combined_report, meter, eval_mode=True)

            # enable train mode again
            self.model.train()

        return combined_report, meter

    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} inference predictions")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm.tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    prepared_batch = to_device(prepared_batch, torch.device("cuda"))
                    with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
                        model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report, self.model)

            logger.info("Finished predicting")
            self.model.train()
    
    def generation_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} image generation")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm.tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    prepared_batch = to_device(prepared_batch, torch.device("cuda"))
                    with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
                        model_output = self.model(prepared_batch)
                        batch_size = prepared_batch["question_id"].size(0)
                        fake_img = model_output["image"]
                        fake_img = fake_img.view(batch_size, -1, fake_img.size(1), fake_img.size(2), fake_img.size(3))
                        caption_num = fake_img.size(1)

                        for batch_num in range(batch_size):
                            for c_num in range(caption_num):
                                im = fake_img[batch_num][c_num].cpu().numpy()
                                # [-1, 1] --> [0, 255]
                                im = (im + 1.0) * 127.5
                                im = im.astype(np.uint8)
                                im = np.transpose(im, (1, 2, 0))
                                im = Image.fromarray(im)
                                img_id = prepared_batch["image_id"][batch_num].cpu().item()
                                quest_id = prepared_batch["question_id"][batch_num].cpu().item()
                                im.save(f"/media/violetta/Data/mmf_data/data/test_output2/{img_id}_{quest_id}_{c_num}.jpg")

                    #report = Report(prepared_batch, model_output)
                    #reporter.add_to_report(report, self.model)

            logger.info("Finished generation")
            self.model.train()
