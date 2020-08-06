# Copyright (c) Facebook, Inc. and its affiliates.
import logging

from mmf.trainers.callbacks.base import Callback
from mmf.utils.checkpoint import Checkpoint

logger = logging.getLogger(__name__)


class CheckpointCallback(Callback):
    """Callback for executing different checkpoint requirements.
    """

    def __init__(self, config, trainer):
        """
        Attr:
            config(mmf_typings.DictConfig): Config for the callback
            trainer(Type[BaseTrainer]): Trainer object
        """
        super().__init__(config, trainer)

        self._checkpoint = Checkpoint(trainer)
        self.checkpoint_interval = self.config.training.checkpoint_interval

    @property
    def checkpoint(self):
        return self._checkpoint

    def on_init_start(self, **kwargs):
        self._checkpoint.load_state_dict()

    def on_batch_end(self, **kwargs):
        if self.trainer.num_updates % self.checkpoint_interval == 0:
            logger.info("Checkpoint time. Saving a checkpoint.")
            self._checkpoint.save(
                self.trainer.num_updates,
                self.trainer.current_iteration,
                update_best=False,
            )

    def on_train_end(self, **kwargs):
        self._checkpoint.restore()
        self._checkpoint.finalize()
