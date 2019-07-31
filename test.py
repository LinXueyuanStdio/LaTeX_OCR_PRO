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
from model.utils.general import Config
from model.utils.text import Vocab
from model.utils.image import greyscale
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
        print(type(img), img.shape, type(formula), len(formula))
        return img, formula

    def __len__(self):
        return len(self.data_generator)


# dir_output = "results/full/"
# config = Config(["configs/data_small.json", "configs/vocab_small.json", "configs/training_small.json", "configs/model.json"])
# config.save(dir_output)
# vocab = Vocab(config)
# val_set = DataGenerator(path_formulas=config.path_formulas_val,
#                         dir_images=config.dir_images_val,
#                         img_prepro=greyscale,
#                         max_iter=config.max_iter,
#                         bucket=config.bucket_val,
#                         path_matching=config.path_matching_val,
#                         max_len=config.max_length_formula,
#                         form_prepro=vocab.form_prepro)
# batch_size = 3
# img_formula_dataset = ImgFormulaDataset(val_set)
# train_loader = torch.utils.data.DataLoader(img_formula_dataset, shuffle=True, num_workers=3, pin_memory=True)
# val_loader = torch.utils.data.DataLoader(img_formula_dataset, shuffle=False, num_workers=3, pin_memory=True)

# for i, (img, formula) in enumerate(minibatches(train_loader, batch_size)):
#     print(i, len(formula[0]),len(formula[1]),len(formula[2]), type(formula), type(formula[0]), type(formula[0][0]))

# print("=====================================================")
# for i, (img, formula) in enumerate(minibatches(val_loader, batch_size)):
#     print(i, len(formula[0]),len(formula[1]),len(formula[2]), type(formula), type(formula[0]), type(formula[0][0]))
