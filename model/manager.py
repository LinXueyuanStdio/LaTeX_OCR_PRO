import PIL
import os
import numpy as np
import time
import threading
from scipy.misc import imread
from PIL import Image

from model.img2seq import Img2SeqModel
from model.utils.general import Config, run
from model.utils.text import Vocab
from model.utils.image import greyscale, crop_image, pad_image, downsample_image, TIMEOUT
from model.utils.visualize_attention import clear_global_attention_slice_stack
from model.utils.visualize_attention import readImageAndShape, vis_attention_slices, getWH
from model.utils.visualize_attention import vis_attention_gif


def getModelForPrediction():
    # restore config and model
    dir_output = "./results/full/"
    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")
    vocab = Vocab(config_vocab)

    model = Img2SeqModel(config_model, dir_output, vocab)
    model.build_pred()
    # model.restore_session(dir_output + "model_weights/model.cpkt")
    return model


def predict_png(model, png_path):
    img = imread(png_path)
    img = greyscale(img)
    hyps = model.predict(img)

    model.logger.info(hyps[0])

    return hyps[0]


class ModelManager(object):
    """ModelManager is a warpper of Model. It extends model and provides more powerful methods."""
    _instance_lock = threading.Lock()  # 线程锁

    def __init__(self, model=None):
        print("init ModelManager")
        if model is None:
            self.model = getModelForPrediction()
        else:
            self.model = model
        print("init model")

    @classmethod
    def instance(cls, *args, **kwargs):
        """多线程安全的单例模式"""
        print("instance")
        if not hasattr(ModelManager, "_instance"):
            with ModelManager._instance_lock:  # 为了保证线程安全在内部加锁
                if not hasattr(ModelManager, "_instance"):
                    ModelManager._instance = ModelManager(*args, **kwargs)
        return ModelManager._instance

    def predict_png(self, png_path):
        """
          Args:
            png_path(string): path to png
          Return:
            latex(string): predicted latex from png

        """
        print("predict_png")
        start = time.time()
        img = imread(png_path)
        img = greyscale(img)
        hyps = self.model.predict(img)
        end = time.time()

        self.model.logger.info(hyps[0])
        print("finish prediction in {}".format(end-start))

        return hyps[0]

    def vis_png(self, png_path):
        dir_output = "./results/full/"
        img_path = png_path
        img, img_w, img_h = readImageAndShape(img_path)
        att_w, att_h = getWH(img_w, img_h)
        print("image path: {0} shape: {1}".format(img_path, (img_w, img_h)))
        clear_global_attention_slice_stack()
        hyps = self.model.predict(img)
        # hyps 是个列表，元素类型是 str, 元素个数等于 beam_search 的 bean_size
        # bean_size 在 `./configs/model.json` 里配置，预训练模型里取 2
        print(hyps[0])

        path_to_save_attention = dir_output+"vis/vis_"+img_path.split('/')[-1][:-4]
        vis_attention_slices(img_path, path_to_save_attention)
        gif_path = vis_attention_gif(img_path, path_to_save_attention, hyps)
        print(gif_path)
        return (hyps[0], gif_path)

    def statistic(self, method):
        print("start")
        print("end")

class DataManager:
    """DataManager is a warpper of data_generator. It provides default implementation."""

    def __init__(self, data_generator):
        super(DataManager, self).__init__()
        self.arg = arg


class Manager(ModelManager):
    """ModelManager is a warpper of Model. It provides default implementation."""

    def __init__(self, data_generator, model):
        super(Manager, self).__init__(model)
        self.data_generator = data_generator
