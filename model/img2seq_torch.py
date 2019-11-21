import time
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn.utils.rnn import pack_padded_sequence

from model.base_torch import BaseModel
from model.utils.general import init_dir, get_logger
from model.utils.general import Progbar
from model.utils.general import Config
from model.utils.general import minibatches
from model.components.SimpleCNN import SimpleCNN
from model.components.ResNet import ResNet9
from model.components.DenseNet import DenseNet169
from model.components.seq2seq_torch import EncoderCNN, DecoderWithAttention, Img2Seq
from model.evaluation.text import score_files, truncate_end, write_answers
from model.utils.image import pad_batch_images_2
from model.utils.text import pad_batch_formulas

from torch.utils.data import Dataset
import h5py
import json
from model.utils.data_generator import DataGenerator


class ImgFormulaDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_generator: DataGenerator, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.data_generator = data_generator
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        (img, formula) = self.data_generator.__getitem__(i)
        img = pad_batch_images_2([img], [800, 800, 1])
        # img = torch.tensor(img, dtype=torch.int8)  # (N, W, H, C)
        # img = img.squeeze(0)
        # img = img.permute(2, 0, 1)  # (C, W, H)
        # if self.transform is not None:
        #     img = self.transform(img)

        # formula = torch.tensor(formula, dtype=torch.int)  # (C, W, H), (TOKEN)
        return img, formula

    def __len__(self):
        return len(self.data_generator)


class Img2SeqModel(BaseModel):
    def __init__(self, config, dir_output, vocab):
        super(Img2SeqModel, self).__init__(config, dir_output)
        self._vocab = vocab

    def getModel(self, model_name="CNN"):
        if model_name == "CNN":
            return SimpleCNN()
        elif model_name == "ResNet9":
            return ResNet9()
        elif model_name == "DenseNet169":
            return DenseNet169(pretrained=True)
        elif model_name == "Img2Seq":
            self.encoder = EncoderCNN(self._config)
            self.decoder = DecoderWithAttention(attention_dim=512,
                                                embed_dim=512,
                                                decoder_dim=512,
                                                vocab_size=self._vocab.n_tok,
                                                dropout=0.5)
            return Img2Seq(self._config, self._vocab)

    def getOptimizer(self, lr_method='adam', lr=0.001):
        self.encoder_optimizer = torch.optim.Adam(params=self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = torch.optim.Adam(params=self.decoder.parameters(), lr=lr)
        return super().getOptimizer(lr_method=lr_method, lr=lr)

    def _run_train_epoch(self, config, train_set, val_set, epoch, lr_schedule):
        """Performs an epoch of training
        Args:
            config: Config instance
            train_set: Dataset instance
            val_set: Dataset instance
            epoch: (int) id of the epoch, starting at 0
            lr_schedule: LRSchedule instance that takes care of learning proc
        Returns:
            score: (float) model will select weights that achieve the highest score
        """
        # logging
        batch_size = config.batch_size
        nbatches = (len(train_set) + batch_size - 1) // batch_size
        prog = Progbar(nbatches)
        self.model.train()
        self.encoder.train()
        self.decoder.train()
        train_loader = torch.utils.data.DataLoader(ImgFormulaDataset(train_set),
                                                   batch_size=batch_size,
                                                   shuffle=True, num_workers=3, pin_memory=True)

        # for i, (img, formula) in enumerate(train_loader):
        for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
            img = pad_batch_images_2(img)
            img = torch.FloatTensor(img)  # (N, W, H, C)
            formula, formula_length = pad_batch_formulas(formula, self._vocab.id_pad, self._vocab.id_end)
            img = img.permute(0, 3, 1, 2)  # (N, C, W, H)
            formula = torch.LongTensor(formula)  # (N,)

            loss_eval = self.getLoss(img, formula=formula, lr=lr_schedule.lr, dropout=config.dropout, training=True)
            prog.update(i + 1, [("loss", loss_eval),  ("lr", lr_schedule.lr)])

            # update learning rate
            lr_schedule.update(batch_no=epoch*nbatches + i)

        self.logger.info("- Training: {}".format(prog.info))
        self.logger.info("- Config: (before evaluate, we need to see config)")
        config.show(fun = self.logger.info)

        # evaluation
        config_eval = Config({"dir_answers": self._dir_output + "formulas_val/", "batch_size": config.batch_size})
        scores = self.evaluate(config_eval, val_set)
        score = scores["perplexity"]
        lr_schedule.update(score=score)

        return score

    def getLoss(self, img, formula, lr, dropout, training=True):
        # Move to GPU, if available
        img = img.to(self.device)
        formula = formula.to(self.device)

        # Forward prop.
        imgs = self.encoder(img)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(
            imgs, formula, torch.LongTensor([[len(i)] for i in formula]))

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = self.criterion(scores, targets)

        alpha_c = 1.
        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        self.decoder_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        loss.backward()

        # Update weights
        self.decoder_optimizer.step()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

        return -loss.item()

    def _run_evaluate_epoch(self, config, test_set):
        """Performs an epoch of evaluation
        Args:
            test_set: Dataset instance
            params: (dict) with extra params in it
                - "dir_name": (string)
        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance
        """
        self.model.eval()
        self.encoder.eval()
        self.decoder.eval()
        # initialize containers of references and predictions
        if self._config.decoding == "greedy":
            refs, hyps = [], [[]]
        elif self._config.decoding == "beam_search":
            refs, hyps = [], [[] for i in range(self._config.beam_size)]
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)
        with torch.no_grad():
            nbatches = len(test_set)
            prog = Progbar(nbatches)
            test_loader = torch.utils.data.DataLoader(ImgFormulaDataset(test_set),
                                                      batch_size=nbatches,
                                                      shuffle=True, num_workers=3, pin_memory=True)

            for i, (img, formula) in enumerate(minibatches(test_set, nbatches)):
                # print(type(img), len(img), img[0].shape)
                # print(type(formula), formula)
                # Move to GPU, if available
                img = pad_batch_images_2(img)
                img = torch.FloatTensor(img)  # (N, W, H, C)
                formula, formula_length = pad_batch_formulas(formula, self._vocab.id_pad, self._vocab.id_end)
                img = img.permute(0, 3, 1, 2)  # (N, C, W, H)
                formula = torch.LongTensor(formula)  # (N,)
                img = img.to(self.device)
                formula = formula.to(self.device)

                # Forward prop.
                imgs = self.encoder(img)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(imgs, formula, torch.LongTensor([[len(i)] for i in formula]))

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                # Calculate loss
                loss = self.criterion(scores, targets)

                print(scores.shape, targets.shape)
                print(loss)

                alpha_c = 1.
                # Add doubly stochastic attention regularization
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                loss_eval = loss.item()

                prog.update(i + 1, [("loss", loss_eval), ("perplexity", np.exp(loss_eval))])

                # Store references (true captions), and hypothesis (prediction) for each image
                # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
                # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
                # print("---------------------------------------------------------------formula and prediction :")
                for form, preds in zip(formula, scores):
                    refs.append(form)
                    # print(form, "    ----------    ", preds[0])
                    for i, pred in enumerate(preds):
                        hyps[i].append(pred)

            files = write_answers(refs, hyps, self._vocab.id_to_tok, config.dir_answers, self._vocab.id_end)
            scores = score_files(files[0], files[1])
            # perp = - np.exp(ce_words / float(n_words))
            # scores["perplexity"] = perp

        self.logger.info("- Evaluating: {}".format(prog.info))

        return {
            "perplexity": loss.item()
        }

    def predict_batch(self, images):
        preds = []
        images = images.to(self.device)
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        pr = outputs[:, 1].detach().cpu().numpy()
        for i in pr:
            preds.append(i)

        return preds

    def predict(self, img):
        return self.predict_batch([img])
