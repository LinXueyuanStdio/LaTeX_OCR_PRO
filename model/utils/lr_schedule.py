import numpy as np


class LRSchedule(object):
    """Class for Learning Rate schedules

    Implements
        - (time) exponential decay with custom range
            - needs to set start_decay, end_decay, lr_init and lr_min
            - set end_decay to None to deactivate
        - (time) warm start:
            - needs to set lr_warm, end_warm.
            - set end_warm to None to deactivate
        - (score) mult decay if no improvement over score
            - needs to set decay_rate
            - set decay_rate to None to deactivate
        - (score) early stopping if no imprv
            - needs to set early_stopping
            - set early_stopping to None to deactivate

    All durations are measured in number of batches
    For usage, must call the update function at each batch.
    You can access the current learning rate with self.lr

    """

    def __init__(self, lr_init=1e-3, lr_min=1e-4, start_decay=0,
        decay_rate=None, end_decay=None, lr_warm=1e-4, end_warm=None,
        early_stopping=None):
        """Initializes Learning Rate schedule

        Sets self.lr and self.stop_training

        Args:
            lr_init: (float) initial lr
            lr_min: (float)
            start_decay: (int) id of batch to start decay
            decay_rate: (float) lr *= decay_rate if no improval. If None, no
                multiplicative decay.
            end_decay: (int) id of batch to end decay. If None, no exp decay
            lr_warm: (float) constant learning rate at the beginning
            end_warm: (int) id of batch to keep the lr_warm before returning to
                lr_init and start the regular schedule.
            early_stopping: (int) number of batches with no imprv

        """
        self._lr_init     = lr_init
        self._lr_min      = lr_min
        self._start_decay = start_decay
        self._decay_rate  = decay_rate
        self._end_decay   = end_decay
        self._lr_warm     = lr_warm
        self._end_warm    = end_warm

        self._score            = None
        self._early_stopping   = early_stopping
        self._n_batch_no_imprv = 0

        # warm start initializes learning rate to warm start
        if self._end_warm is not None:
            # make sure that decay happens after the warm up
            self._start_decay = max(self._end_warm, self._start_decay)
            self.lr = self._lr_warm
        else:
            self.lr = lr_init

        # setup of exponential decay
        if self._end_decay is not None:
            self._exp_decay = np.power(lr_min/lr_init, 1/float(self._end_decay - self._start_decay))


    @property
    def stop_training(self):
        """For Early Stopping"""
        if (self._early_stopping is not None and
            (self._n_batch_no_imprv >= self._early_stopping)):
            return True
        else:
            return False


    def update(self, batch_no=None, score=None):
        """Updates the learning rate

        (score) decay by self.decay rate if score is higher than previous
        (time) update lr according to
            - warm up
            - exp decay
        Both updates can concurrently happen

        Args:
            batch_no: (int) id of the batch
            score: (float) score, higher is better

        """
        # update based on time
        if batch_no is not None:
            if (self._end_warm is not None and
                (self._end_warm <= batch_no <= self._start_decay)):
                self.lr = self._lr_init

            if batch_no > self._start_decay and self._end_decay is not None:
                self.lr *= self._exp_decay

        # update based on performance
        if self._decay_rate is not None:
            if score is not None and self._score is not None:
                if score <= self._score:
                    self.lr *= self._decay_rate
                    self._n_batch_no_imprv += 1
                else:
                    self._n_batch_no_imprv = 0

        # update last score eval
        if score is not None:
            self._score = score

        self.lr = max(self.lr, self._lr_min)
