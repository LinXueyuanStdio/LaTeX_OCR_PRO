import os
import time

import torch
import torch.nn as nn

from model.utils.general import init_dir, get_logger


class BaseModel(object):
    """Generic class for our model
    Usage:

    1. init
    2. build_train() or build_pred()
    3. save and restore
    4. train and evaluate
    """
    # 1. init

    def __init__(self, config, dir_output):
        """Defines self._config
        Args:
            config: (Config instance) class with hyper parameters, from "model.json"
            dir_output: output dir
        """
        self._config = config
        self._dir_output = dir_output
        self._init_relative_path(dir_output)
        self.logger = get_logger(dir_output + "model.log")

    def _init_relative_path(self, dir_output):
        # init parent dir
        init_dir(dir_output)

        # 1. init child dir
        # check dir one last time
        self._dir_model = dir_output + "model_weights/"
        init_dir(self._dir_model)

        # 2. define model path
        self._model_path = self._dir_model+"model.cpkt"

    # 2. build
    def build_train(self, config=None):
        """To overwrite with model-specific logic
        This logic must define
            - self.model_name
            - self.loss
            - self.lr
            - etc.
        Args:
            config: from "training.json" and "model.json"
        """
        self.logger.info("- Building model...")

        self._init_model(config.model_name, config.device)
        self._init_optimizer(config.lr_method, config.lr_init)
        self._init_scheduler(config.lr_scheduler)
        self._init_criterion(config.criterion_method)

        self.logger.info("- done.")

    def build_pred(self, config=None):
        """Similar to build_train but no need to define train_op
        Args:
            config: from "model.json"
        """
        self.logger.info("- Building model...")
        self._init_model(config.model_name, config.device)
        self.logger.info("- done.")

    def _init_model(self, model_name="CNN", device="cpu"):
        self.logger.info("   - " + model_name)
        self.logger.info("   - " + device)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.getModel(model_name)
        self.model = self.model.to(self.device)

    def _init_optimizer(self, lr_method="adam", lr=1e-3):
        """Defines self.optimizer that performs an update on a batch
        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: init learning rate (initial value)
        """
        # 1. optimizer
        _lr_m = lr_method.lower()  # lower to make sure
        print("  - " + _lr_m)
        self.optimizer = self.getOptimizer(_lr_m, lr)

    def _init_scheduler(self, lr_scheduler="CosineAnnealingLR"):
        """Defines self.scheduler that performs an update on a batch
        Args:
            lr_scheduler: (string) learning rate schedule method, for example "CosineAnnealingLR"
        """
        # 2. scheduler
        print("  - lr_scheduler " + lr_scheduler)
        self.scheduler = self.getLearningRateScheduler(lr_scheduler)

    def _init_criterion(self, criterion_method="CrossEntropyLoss"):
        """Defines self.criterion that performs an update on a batch
        Args:
            criterion_method: (string) criterion method, for example "CrossEntropyLoss"
        """
        # 3. criterion
        print("  - " + criterion_method)
        self.criterion = self.getCriterion(criterion_method)

    # ! MUST OVERWRITE
    def getModel(self, model_name="CNN"):
        """return your Model
        Args:
            model_name: String, from "model.json"
        Returns:
            your model that inherits from torch.nn
        """
        raise NotImplementedError("return your model ({}) that inherits from torch.nn".format(model_name))

    def getOptimizer(self, lr_method="adam", lr=1e-3):
        if lr_method == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif lr_method == 'adamax':
            return torch.optim.Adamax(self.model.parameters(), lr=lr)
        elif lr_method == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Unknown Optimizer {}".format(lr_method))

    def getLearningRateScheduler(self, lr_scheduler="CosineAnnealingLR"):
        if lr_scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=4e-08)
        else:
            raise NotImplementedError("Unknown Learning Rate Scheduler {}".format(lr_scheduler))

    def getCriterion(self, criterion_method="CrossEntropyLoss"):
        if criterion_method == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss()
        elif criterion_method == 'MSELoss':
            return torch.nn.MSELoss()
        elif criterion_method == 'BCEWithLogitsLoss':
            return torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("Unknown Criterion Method {}".format(criterion_method))

    # 3. save and restore
    def auto_restore(self):
        if os.path.exists(self._model_path) and os.path.isfile(self._model_path):
            self.restore()

    def restore(self, model_path=None, map_location='cpu'):
        """Reload weights into session
        Args:
            model_path: weights path "model_weights/model.cpkt"
            map_location: 'cpu' or 'gpu:0'
        """
        self.logger.info("- Reloading the latest trained model...")
        if model_path == None:
            self.model.load_state_dict(torch.load(self._model_path, map_location=map_location))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=map_location))

    def save(self):
        """Saves model"""
        self.logger.info("- Saving model...")
        torch.save(self.model.state_dict(), self._model_path)
        self.logger.info("- Saved model in {}".format(self._dir_model))

    # 4. train and evaluate
    def train(self, config, train_set, val_set, lr_schedule):
        """Global training procedure
        Calls method self.run_epoch and saves weights if score improves.
        All the epoch-logic including the lr_schedule update must be done in
        self.run_epoch
        Args:
            config: Config instance contains params as attributes
            train_set: Dataset instance
            val_set: Dataset instance
            lr_schedule: LRSchedule instance that takes care of learning proc
            path_label: dataframe
        Returns:
            best_score: (float)
        """
        best_score = None

        for epoch in range(config.n_epochs):
            # logging
            tic = time.time()
            self.logger.info("Epoch {:}/{:}".format(epoch+1, config.n_epochs))

            # epoch
            score = self._run_train_epoch(config, train_set, val_set, epoch, lr_schedule)

            # save weights if we have new best score on eval
            if best_score is None or score >= best_score:  # abs(score-0.5) <= abs(best_score-0.5):
                best_score = score
                self.logger.info("- New best score ({:04.2f})!".format(best_score))
                self.save()
            if lr_schedule.stop_training:
                self.logger.info("- Early Stopping.")
                break

            # logging
            toc = time.time()
            self.logger.info("- Elapsed time: {:04.2f}, learning rate: {:04.5f}".format(toc-tic, lr_schedule.lr))

        return best_score

    def evaluate(self, config, test_set):
        """Evaluates model on test set
        Calls method run_evaluate on test_set and takes care of logging
        Args:
            config: Config
            test_set: instance of class Dataset
            path_label: dataframe
        Return:
            scores: (dict) scores["acc"] = 0.85 for instance
        """
        self.logger.info("- Evaluating...")
        scores = self._run_evaluate_epoch(config, test_set)  # evaluate
        msg = " || ".join([" {} is {:04.2f} ".format(k, v) for k, v in scores.items()])
        self.logger.info("- Eval: {}".format(msg))

        return scores

    def _auto_backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ! MUST OVERWRITE
    def _run_train_epoch(config, train_set, val_set, epoch, lr_schedule):
        """Model_specific method to overwrite
        Performs an epoch of training
        Args:
            config: Config
            train_set: Dataset instance
            val_set: Dataset instance
            epoch: (int) id of the epoch, starting at 0
            lr_schedule: LRSchedule instance that takes care of learning proc
        Returns:
            score: (float) model will select weights that achieve the highest score
        Alert:
            you can use the method below to simplify your code
            _auto_backward(self, loss)
        """
        raise NotImplementedError("Performs an epoch of training")

    # ! MUST OVERWRITE
    def _run_evaluate_epoch(config, test_set):
        """Model-specific method to overwrite
        Performs an epoch of evaluation
        Args:
            config: Config
            test_set: Dataset instance
        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance
        """
        raise NotImplementedError("Performs an epoch of evaluation")
