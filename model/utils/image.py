import os
import numpy as np
import os
import PIL
from PIL import Image
from multiprocessing import Pool


from .general import run, get_files, delete_file, init_dir


TIMEOUT = 10


def get_max_shape(arrays):
    """
    Args:
        images: list of arrays

    """
    # hack
    # return [40, 240, 1]
    shapes = list(map(lambda x: list(x.shape), arrays))
    return [max(x) for x in zip(*shapes)]


def pad_batch_images(images, max_shape=None):
    """
    Args:
        images: list of arrays
        target_shape: shape at which we want to pad

    """

    # 1. max shape
    if max_shape is None:
        max_shape = get_max_shape(images)

    # 2. apply formating
    batch_images = 255 * np.ones([len(images)] + list(max_shape))
    for idx, img in enumerate(images):
        batch_images[idx, :img.shape[0], :img.shape[1]] = img

    return batch_images.astype(np.uint8)


def pad_batch_images_2(images, max_shape=None):
    """
    Args:
        images: list of arrays
        target_shape: shape at which we want to pad

    """

    # 1. max shape
    if max_shape is None:
        max_shape = get_max_shape(images)

    # 2. apply formating
    batch_images = 255 * np.ones([len(images)] + list(max_shape))

    for idx, img in enumerate(images):
        batch_images[idx, :img.shape[0], :img.shape[1]] = img

    return batch_images.astype(np.uint8)

def greyscale(state):
    """Preprocess state (:, :, 3) image into greyscale"""
    state = state[:, :, 0]*0.299 + state[:, :, 1]*0.587 + state[:, :, 2]*0.114
    state = state[:, :, np.newaxis]
    return state.astype(np.uint8)


def downsample(state):
    """Downsamples an image on the first 2 dimensions

    Args:
        state: (np array) with 3 dimensions

    """
    return state[::2, ::2, :]


def pad_image(img, output_path, pad_size=[8, 8, 8, 8], buckets=None):
    """Pads image with pad size and with buckets

    Args:
        img: (string) path to image
        output_path: (string) path to output image
        pad_size: list of 4 ints
        buckets: ascending ordered list of sizes, [(width, height), ...]

    """
    top, left, bottom, right = pad_size
    old_im = Image.open(img)
    old_size = (old_im.size[0] + left + right, old_im.size[1] + top + bottom)
    new_size = get_new_size(old_size, buckets)
    new_im = Image.new("RGB", new_size, (255, 255, 255))
    new_im.paste(old_im, (left, top))
    new_im.save(output_path)


def get_new_size(old_size, buckets):
    """Computes new size from buckets

    Args:
        old_size: (width, height)
        buckets: list of sizes

    Returns:
        new_size: original size or first bucket in iter order that matches the
            size.

    """
    if buckets is None:
        return old_size
    else:
        w, h = old_size
        for (w_b, h_b) in buckets:
            if w_b >= w and h_b >= h:
                return w_b, h_b

        return old_size


def crop_image(img, output_path):
    """Crops image to content

    Args:
        img: (string) path to image
        output_path: (string) path to output image

    """
    old_im = Image.open(img).convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        old_im.save(output_path)
        return False

    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
    old_im.save(output_path)
    return True


def downsample_image(img, output_path, ratio=2):
    """Downsample image by ratio"""
    assert ratio >= 1, ratio
    if ratio == 1:
        return True
    old_im = Image.open(img)
    old_size = old_im.size
    new_size = (int(old_size[0]/ratio), int(old_size[1]/ratio))

    new_im = old_im.resize(new_size, PIL.Image.LANCZOS)
    new_im.save(output_path)
    return True


def convert_to_png(formula, dir_output, name, quality=100, density=200,
                   down_ratio=2, buckets=None):
    """Converts LaTeX to png image

    Args:
        formula: (string) of latex
        dir_output: (string) path to output directory
        name: (string) name of file
        down_ratio: (int) downsampling ratio
        buckets: list of tuples (list of sizes) to produce similar shape images

    """
    # write formula into a .tex file
    with open(dir_output + "{}.tex".format(name), "w") as f:
        f.write(r"""\documentclass[preview]{standalone}
    \begin{document}
        $$ %s $$
    \end{document}""" % (formula))

    # call pdflatex to create pdf
    run("pdflatex -interaction=nonstopmode -output-directory={} {}".format(
        dir_output, dir_output+"{}.tex".format(name)), TIMEOUT)

    # call magick to convert the pdf into a png file
    run("magick convert -density {} -quality {} {} {}".format(density, quality,
                                                              dir_output+"{}.pdf".format(name),
                                                              dir_output+"{}.png".format(name)),
        TIMEOUT)

    # cropping and downsampling
    img_path = dir_output + "{}.png".format(name)

    try:
        crop_image(img_path, img_path)
        pad_image(img_path, img_path, buckets=buckets)
        downsample_image(img_path, img_path, down_ratio)
        clean(dir_output, name)

        return "{}.png".format(name)

    except Exception as e:
        print(e)
        clean(dir_output, name)
        return False


def clean(dir_output, name):
    delete_file(dir_output+"{}.aux".format(name))
    delete_file(dir_output+"{}.log".format(name))
    delete_file(dir_output+"{}.pdf".format(name))
    delete_file(dir_output+"{}.tex".format(name))


def build_image(item):
    idx, form, dir_images, quality, density, down_ratio, buckets = item
    name = str(idx)
    path_img = convert_to_png(form, dir_images, name, quality, density,
                              down_ratio, buckets)
    return (path_img, idx)


def build_images(formulas, dir_images, quality=100, density=200, down_ratio=2, buckets=None, n_threads=4):
    """Parallel procedure to produce images from formulas

    If some of the images have already been produced, does not recompile them.

    Args:
        formulas: (dict) idx -> string

    Returns:
        list of (path_img, idx). If an exception was raised during the image
            generation, path_img = False
    """
    init_dir(dir_images)
    existing_idx = sorted(set([int(file_name.split('.')[0])
                               for file_name in get_files(dir_images)
                               if file_name.split('.')[-1] == "png"]))

    pool = Pool(n_threads)
    result = pool.map(build_image, [(idx, form, dir_images, quality, density, down_ratio, buckets)
                                    for idx, form in formulas.items()
                                    if idx not in existing_idx])
    pool.close()
    pool.join()

    result += [(str(idx) + ".png", idx) for idx in existing_idx]

    return result
