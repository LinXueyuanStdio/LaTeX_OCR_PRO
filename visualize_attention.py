import sys
import click
import numpy as np
import PIL.Image as PILImage
import os
import PIL
from scipy.misc import imread
from matplotlib import transforms
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.image as mpimg  # mpimg 用于读取图片
import matplotlib.pyplot as plt  # plt 用于显示图片

from model.utils.image import greyscale, crop_image, pad_image, downsample_image, TIMEOUT
from model.utils.text import Vocab
from model.utils.general import Config, run
from model.img2seq import Img2SeqModel
import model.components.attention_mechanism


def getWH(img_w, img_h):
    '''
    模仿卷积层 encoder 缩放
    '''
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w - 2), np.ceil(img_h - 2)
    return int(img_w), int(img_h)


def vis_attention_slices(img_path, path_to_save_attention):
    '''
    可视化所有的 attention slices，保存为 png
    '''
    img, img_w, img_h = readImageAndShape(img_path)
    att_w, att_h = getWH(img_w, img_h)
    for i in range(len(model.components.attention_mechanism.ctx_vector)):
        attentionVector = model.components.attention_mechanism.ctx_vector[i]
        filename = getFileNameToSave(path_to_save_attention, i)
        vis_attention_slice(attentionVector, img_path, filename, img_w, img_h, att_w, att_h)


def getFileNameToSave(path_to_save_attention, i):
    return path_to_save_attention+"_"+str(i)+".png"


def getOutArray(attentionVector, att_w, att_h):
    '''
    将 attentionVector 重新构建为宽 att_w, 高 att_h 的图片矩阵
    '''
    att = sorted(list(enumerate(attentionVector[0].flatten())),
                 key=lambda tup: tup[1],
                 reverse=True)  # attention 按权重从大到小递减排序
    idxs, att = zip(*att)

    positions = idxs[:]

    # 把扁平化的一维的 attention slice 重整成二维的图片矩阵，像素颜色值范围 [0, 255]
    outarray = np.ones((att_h, att_w)) * 255.

    for i in range(len(positions)):
        pos = positions[i]
        loc_x = int(pos / att_w)
        loc_y = int(pos % att_w)
        att_pos = att[i]
        outarray[loc_x, loc_y] = (1 - att_pos) * 255.
        # (1 - att_pos) * 255. 而不是直接 att_pos * 255
        # 因为颜色值越小越暗，而权重需要越大越暗

    return outarray


def vis_attention_slice(attentionVector, img_path, path_to_save_attention, img_w, img_h, att_w, att_h):
    combine = getCombineArray(attentionVector, img_path, img_w, img_h, att_w, att_h)
    plt.figure()
    plt.imsave(path_to_save_attention, combine)


def getCombineArray(attentionVector, img_path, img_w, img_h, att_w, att_h):
    outarray = getOutArray(attentionVector, att_w, att_h)

    out_image = PILImage.fromarray(outarray).resize((img_w, img_h), PILImage.NEAREST)
    inp_image = PILImage.open(img_path)
    combine = PILImage.blend(inp_image.convert('RGBA'), out_image.convert('RGBA'), 0.5)
    return np.asarray(combine)


def rainbow_text(x, y, strings, colors, ax=None, **kw):
    """
    Take a list of ``strings`` and ``colors`` and place them next to each
    other, with text strings[i] being shown in colors[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.

    The text will get added to the ``ax`` axes, if provided, otherwise the
    currently active axes will be used.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    # horizontal version
    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kw)
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(
            text.get_transform(), x=ex.width, units='dots')


def vis_attention_gif(img_path, path_to_save_attention, hyps, full_latex=False):
    img, img_w, img_h = readImageAndShape(img_path)
    att_w, att_h = getWH(img_w, img_h)
    LaTeX_symbols = hyps[0].split(" ")
    LaTeX_symbols_count = len(LaTeX_symbols)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    fig.set_figwidth(25)
    fig.set_figheight(6)

    # 询问图形在屏幕上的大小和DPI（每英寸点数）
    # 注意当把图形保存为文件时，需要为此单独再提供一个DPI
    print('fig size: {0} DPI, size in inches {1}'.format(fig.get_dpi(), fig.get_size_inches()))
    print('processing...')

    def update(i):
        '''
        在这里绘制动画帧
        args:
        i : (int) range [0, ?)
        return:
        (tuple) 以元组形式返回这一帧需要重新绘制的物体
        '''
        # 1. 更新标题
        LaTeX_symbols_colors = ['green']*LaTeX_symbols_count
        if i < LaTeX_symbols_count:
            LaTeX_symbols_colors[i] = "red"
        symbol_count = LaTeX_symbols_count if full_latex else i+1
        rainbow_text(0, img_h+6, LaTeX_symbols[:symbol_count], LaTeX_symbols_colors[:symbol_count], ax, size=36)

        # 2. 更新图片
        attentionVector = model.components.attention_mechanism.ctx_vector[i]
        combine = getCombineArray(attentionVector, img_path, img_w, img_h, att_w, att_h)

        ax.imshow(np.asarray(combine))
        # 3. 以元组形式返回这一帧需要重新绘制的物体
        return ax

    # 会为每一帧调用Update函数
    # 这里FunAnimation设置一个10帧动画，每帧间隔200ms
    plt.title("Visualize Attention over Image", fontsize=40)
    anim = FuncAnimation(fig, update, frames=np.arange(0, LaTeX_symbols_count), interval=200)
    anim.save(path_to_save_attention+'_visualization1.gif', dpi=80, writer='imagemagick')
    print("finish!")


def clear_global_attention_slice_stack():
    '''
    这是 attention 的全局变量
    务必在调用 img2SeqModel.predict() 之前把 attention slices 栈清空
    不然每预测一次，各自不同公式的 attention slices 会堆在一起
    '''
    model.components.attention_mechanism.need_to_export = True
    model.components.attention_mechanism.ctx_vector = []


def readImageAndShape(img_path):
    lena = mpimg.imread(img_path)  # 读取目录下的图片，返回 np.array
    img = imread(img_path)
    img = greyscale(img)

    img_w, img_h = lena.shape[1], lena.shape[0]

    return img, img_w, img_h


def vis_img_with_attention(img2SeqModel, img_path, dir_output):
    clear_global_attention_slice_stack()
    img, img_w, img_h = readImageAndShape(img_path)
    att_w, att_h = getWH(img_w, img_h)
    print("image path: {0} shape: {1}".format(img_path, (img_w, img_h)))
    hyps = img2SeqModel.predict(img)
    # hyps 是个列表，元素类型是 str, 元素个数等于 beam_search 的 bean_size
    # bean_size 在 `./configs/model.json` 里配置，预训练模型里取 2
    print(hyps[0])

    path_to_save_attention = dir_output+"vis/vis_"+img_path.split('/')[-1][:-4]
    vis_attention_slices(img_path, path_to_save_attention)
    vis_attention_gif(img_path, path_to_save_attention, hyps)


@click.command()
@click.option('--image', default="data/images_test/6.png",
              help='Path to image to OCR')
@click.option('--vocab', default="configs/vocab.json",
              help='Path to vocab json config')
@click.option('--model', default="configs/model.json",
              help='Path to model json config')
@click.option('--output', default="results/full/",
              help='Dir for results and model weights')
def main(image, vocab, model, output):
    dir_output = output
    img_path = image

    # 加载配置，根据配置初始化字典和模型
    config = Config([vocab, model])
    vocab = Vocab(config)
    img2SeqModel = Img2SeqModel(config, dir_output, vocab)
    img2SeqModel.build_pred()

    vis_img_with_attention(img2SeqModel, img_path, dir_output)


if __name__ == "__main__":
    main()
