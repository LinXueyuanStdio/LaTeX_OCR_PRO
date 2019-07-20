import time
import os
import numpy as np
from scipy.misc import imread


from .text import load_formulas
from .image import build_images, greyscale
from .general import init_dir


class DataGeneratorFile(object):
    """Simple Generator of tuples (img_path, formula_id)"""

    def __init__(self, filename):
        """Inits Data Generator File

        Iterator that returns
            tuple (img_path, formula_id)

        Args:
            filename: (string of path to file)

        """
        self._filename = filename

    def __iter__(self):
        with open(self._filename) as f:
            for line in f:
                line = line.strip().split(" ")
                path_img, id_formula = line[0], line[1]
                yield path_img, id_formula


class DataGenerator(object):
    """Data Generator of tuple (image, formula)"""

    def __init__(self, path_formulas, dir_images, path_matching, bucket=False,
                 form_prepro=lambda s: s.strip().split(" "), iter_mode="data",
                 img_prepro=lambda x: x, max_iter=None, max_len=None,
                 bucket_size=20):
        """Initializes the DataGenerator

        Args:
            path_formulas: (string) file of formulas.
            dir_images: (string) dir of images, contains jpg files.
            path_matching: (string) file of name_of_img, id_formula
            img_prepro: (lambda function) takes an array -> an array. Default,
                identity
            form_prepro: (lambda function) takes a string -> array of int32.
                Default, identity.
            max_iter: (int) maximum numbers of elements in the dataset
            max_len: (int) maximum length of a formula in the dataset
                if longer, not yielded.
            iter_mode: (string) "data", "full" to set the type returned by the
                generator
            bucket: (bool) decides if bucket the data by size of image
            bucket_size: (int)

        """
        self._path_formulas = path_formulas
        self._dir_images = dir_images
        self._path_matching = path_matching
        self._img_prepro = img_prepro
        self._form_prepro = form_prepro
        self._max_iter = max_iter
        self._max_len = max_len
        self._iter_mode = iter_mode
        self._bucket = bucket
        self._bucket_size = bucket_size

        self._length = None
        self._formulas = self._load_formulas(path_formulas)

        self._set_data_generator()

    def _set_data_generator(self):
        """Sets iterable or generator of tuples (img_path, id of formula)"""
        self._data_generator = DataGeneratorFile(self._path_matching)

        if self._bucket:
            self._data_generator = self.bucket(self._bucket_size)

    def bucket(self, bucket_size):
        """Iterates over the listing and creates buckets of same shape images.

        Args:
            bucket_size: (int) size of the bucket

        Returns:
            bucketed_dataset: [(img_path1, id1), ...]

        """
        print("Bucketing the dataset...")
        bucketed_dataset = []
        old_mode = self._iter_mode  # store the old iteration mode
        self._iter_mode = "full"

        # iterate over the dataset in "full" mode and create buckets
        data_buckets = dict()  # buffer for buckets
        for idx, (img, formula, img_path, formula_id) in enumerate(self):
            s = img.shape
            if s not in data_buckets:
                data_buckets[s] = []
            # if bucket is full, write it and empty it
            if len(data_buckets[s]) == bucket_size:
                for (img_path, formula_id) in data_buckets[s]:
                    bucketed_dataset += [(img_path, formula_id)]
                data_buckets[s] = []

            data_buckets[s] += [(img_path, formula_id)]

        # write the rest of the buffer
        for k, v in data_buckets.items():
            for (img_path, formula_id) in v:
                bucketed_dataset += [(img_path, formula_id)]

        self._iter_mode = old_mode
        self._length = idx + 1

        print("- done.")
        return bucketed_dataset

    def _load_formulas(self, filename):
        """Loads txt file with formulas in a dict

        Args:
            filename: (string) path of formulas.

        Returns:
            dict: dict[idx] = one formula

        """
        formulas = load_formulas(filename)
        return formulas

    def _get_raw_formula(self, formula_id):
        try:
            formula_raw = self._formulas[int(formula_id)]
        except KeyError:
            print("Tried to access id {} but only {} formulas".format(formula_id, len(self._formulas)))
            print("Possible fix: mismatch between matching file and formulas")
            raise KeyError

        return formula_raw

    def _process_instance(self, example):
        """From path and formula id, returns actual data

        Applies preprocessing to both image and formula

        Args:
            example: tuple (img_path, formula_ids)
                img_path: (string) path to image
                formula_id: (int) id of the formula

        Returns:
            img: depending on _img_prepro
            formula: depending on _form_prepro

        """
        img_path, formula_id = example

        img = imread(self._dir_images + img_path)
        img = self._img_prepro(img)
        formula = self._form_prepro(self._get_raw_formula(formula_id))  # py3.x 要加 list()， 不然会返回 map

        if self._iter_mode == "data":
            inst = (img, formula)
        elif self._iter_mode == "full":
            inst = (img, formula, img_path, formula_id)

        # filter on the formula length
        if self._max_len is not None and len(formula) > self._max_len:
            skip = True
        else:
            skip = False

        return inst, skip

    def __iter__(self):
        """Iterator over Dataset

        Yields:
            tuple (img, formula)

        """
        n_iter = 0
        for example in self._data_generator:
            if self._max_iter is not None and n_iter >= self._max_iter:
                break
            result, skip = self._process_instance(example)
            if skip:
                continue
            n_iter += 1
            yield result

    def __getitem__(self, i):
        counter = 0
        for item in self:
            if counter == i:
                return item
            counter += 1
        raise NotImplementedError("IndexOutOfBound")

    def __len__(self):
        if self._length is None:
            print("First call to len(dataset) - may take a while.")
            counter = 0
            for _ in self:
                counter += 1
            self._length = counter
            print("- done.")

        return self._length

    def build(self, quality=100, density=200, down_ratio=2, buckets=None, n_threads=10):
        """Generates images from the formulas and writes the correspondance
        in the matching file.

        Args:
            quality: parameter for magick
            density: parameter for magick
            down_ratio: (int) downsampling ratio
            buckets: list of tuples (list of sizes) to produce similar
                shape images

        """
        # 1. produce images
        init_dir(self._dir_images)
        result = build_images(self._formulas, self._dir_images, quality,
                              density, down_ratio, buckets, n_threads)

        # 2. write matching with same convention of naming
        with open(self._path_matching, "w") as f:
            for (path_img, idx) in result:
                if path_img is not False:  # image was successfully produced
                    f.write("{} {}\n".format(path_img, idx))
