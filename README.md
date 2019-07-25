# LaTeX_OCR_PRO

数学公式识别，增强：中文公式、手写公式

![](https://raw.githubusercontent.com/LinXueyuanStdio/LaTeX_OCR/master/art/6.png)
![](https://raw.githubusercontent.com/LinXueyuanStdio/LaTeX_OCR/master/art/visualization_6_short.gif)
![](https://raw.githubusercontent.com/LinXueyuanStdio/LaTeX_OCR/master/art/12.png)
![](https://raw.githubusercontent.com/LinXueyuanStdio/LaTeX_OCR/master/art/visualization_12_short.gif)
![](https://raw.githubusercontent.com/LinXueyuanStdio/LaTeX_OCR/master/art/14.png)
![](https://raw.githubusercontent.com/LinXueyuanStdio/LaTeX_OCR/master/art/visualization_14_short.gif)

Seq2Seq + Attention + Beam Search。结构如下：

![](https://raw.githubusercontent.com/LinXueyuanStdio/LaTeX_OCR/master/art/architecture.jpg)

* [1. 搭建环境](#1-搭建环境)
* [2. 开始训练](#2-开始训练)
* [3. 可视化](#3-可视化)
* [4. 评价](#4-评价)
* [5. 模型的具体实现细节](#5-模型的具体实现细节)
    * [总述](#总述)
    * [数据获取和数据处理](#数据获取和数据处理)
    * [模型构建](#模型构建)
* [6. 踩坑记录](#6-踩坑记录)
    * [win10 用 GPU 加速训练](#win10-用-gpu-加速训练)
    * [如何可视化Attention层](#如何可视化attention层)
* [致谢](#致谢)

## 1. 搭建环境

1. python3.5 + tensorflow1.12.2
2. latex (latex 转 pdf)
3. ghostscript (图片处理)
4. magick (pdf 转 png)

<details>
  <summary>Linux</summary>

一键安装
```shell
make install-linux
```
或
1. 安装本项目依赖
```shell
virtualenv env35 --python=python3.5
source env35/bin/activate
pip install -r requirements.txt
```
2. 安装 latex (latex 转 pdf)
```shell
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-latex-extra
```
3. 安装 ghostscript
```shell
sudo apt-get update
sudo apt-get install ghostscript
sudo apt-get install libgs-dev
```
4. 安装[magick](https://www.imagemagick.org/script/install-source.php) (pdf 转 png)
```shell
wget http://www.imagemagick.org/download/ImageMagick.tar.gz
tar -xvf ImageMagick.tar.gz
cd ImageMagick-7.*; \
./configure --with-gslib=yes; \
make; \
sudo make install; \
sudo ldconfig /usr/local/lib
rm ImageMagick.tar.gz
rm -r ImageMagick-7.*
```
</details>

<details>
  <summary>Mac</summary>

一键安装

```shell
make install-mac
```

或
1. 安装本项目依赖
```shell
sudo pip install -r requirements.txt
```
2. LaTeX 请自行安装

3. 安装[magick](https://www.imagemagick.org/script/install-source.php) (pdf 转 png)

```shell
wget http://www.imagemagick.org/download/ImageMagick.tar.gz
tar -xvf ImageMagick.tar.gz
cd ImageMagick-7.*; \
./configure --with-gslib=yes; \
make;\
sudo make install; \
rm ImageMagick.tar.gz
rm -r ImageMagick-7.*
```

</details>

## 2. 开始训练


<details>
  <summary>生成小数据集、训练、评价</summary>

提供了样本量为 100 的小数据集，方便测试。只需 2 分钟就可以根据 `./data/small.formulas/` 下的公式生成用于训练的图片。

一步训练

```
make small
```
或

1. 生成数据集

   用 LaTeX 公式生成图片，同时保存公式-图片映射文件，生成字典 __只用运行一次__

    ```shell
    # 默认
    python build.py
    # 或者
    python build.py --data=configs/data_small.json --vocab=configs/vocab_small.json
    ```

2. 训练
    ```
    # 默认
    python train.py
    # 或者
    python train.py --data=configs/data_small.json --vocab=configs/vocab_small.json --training=configs/training_small.json --model=configs/model.json --output=results/small/
    ```

3. 评价预测的公式
    ```
    # 默认
    python evaluate_txt.py
    # 或者
    python evaluate_txt.py --results=results/small/
    ```

4. 评价数学公式图片

    ```
    # 默认
    python evaluate_img.py
    # 或者
    python evaluate_img.py --results=results/small/
    ```

</details>

<details>
  <summary>生成完整数据集、训练、评价</summary>

根据公式生成 70,000+ 数学公式图片需要 `2`-`3` 个小时

一步训练

```
make full
```
或

1. 生成数据集

   用 LaTeX 公式生成图片，同时保存公式-图片映射文件，生成字典 __只用运行一次__
    ```
    python build.py --data=configs/data.json --vocab=configs/vocab.json
    ```

2. 训练
    ```
    python train.py --data=configs/data.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/full/
    ```

3. 评价预测的公式
    ```
    python evaluate_txt.py --results=results/full/
    ```

4. 评价数学公式图片
    ```
    python evaluate_img.py --results=results/full/
    ```

</details>

## 3. 可视化

<details>
  <summary>可视化训练过程</summary>

用 tensorboard 可视化训练过程

小数据集

```
cd results/small
tensorboard --logdir ./
```

完整数据集

```
cd results/full
tensorboard --logdir ./
```
</details>

<details>
  <summary>可视化预测过程</summary>

打开 `visualize_attention.ipynb`，一步步观察模型是如何预测 LaTeX 公式的。

或者运行

```shell
# 默认
python visualize_attention.py
# 或者
python visualize_attention.py --image=data/images_test/6.png --vocab=configs/vocab.json --model=configs/model.json --output=results/full/
```

可在 `--output` 下生成预测过程的注意力图。

</details>

## 4. 评价

|      指标       | 训练分数 | 测试分数 |
| :-------------: | :------: | :------: |
|   perplexity    |   1.39   |   1.44   |
|  EditDistance   |  81.68   |  80.45   |
|     BLEU-4      |  78.21   |  75.42   |
| ExactMatchScore |  13.93   |  12.44   |

perplexity 是越接近1越好，其余3个指标是越大越好。ExactMatchScore 比较低，继续训练应该可以到 70 以上。机器不太好，训练太费时间了。

## 5. 部署

1. 安装部署需要的环境
   ```bash
   pip install django
   ```
2. 开启服务
   ```bash
   python manage.py runserver 0.0.0.0:8010
   ```
3. 开启图片服务
   ```bash
   cd data/images_train
   python -m SimpleHTTPServer 8020
   ```
4. 使用方法
   在输入框里依次输入 `0.png`, `1.png` 等等，即可看到结果


## 5. 模型的具体实现细节

### 总述

首先我们获取到足够的公式，对公式进行规范化处理，方便划分出字典。然后通过规范化的公式使用脚本生成图片，具体用到了latex和ghostscript和magick，同时保存哪个公式生成哪个图片，保存为公式-图片映射文件。这样我们得到了3个数据集：规范化的公式集，图片集，公式-图片映射集，还有个附赠品：latex字典。这个字典决定了模型的上限，也就是说，模型预测出的公式只能由字典里的字符组成，不会出现字典以外的字符。

然后构建模型。

模型分为3部分，数据生成器，神经网络模型，使用脚本。

数据生成器读取公式-图片映射文件，为模型提供(公式, 图片)的矩阵元组。

神经网络模型是 Seq2Seq + Attention + Beam Search。Seq2Seq的Encoder是CNN，Decoder是LSTM。Encoder和Decoder之间插入Attention层，具体操作是这样：Encoder到Decoder有个扁平化的过程，Attention就是在这里插入的。随Attention插入的还有我们自定义的一个op，用来导出Attention的数据，做Attention的可视化。

使用脚本包括构建脚本、训练脚本、测试脚本、预测脚本、评估脚本、可视化脚本。使用说明看上面的命令行就行。

训练过程根据epoch动态调整LearningRate。decoder可以选择用`lstm`或`gru`，在`configs/model.json`里改就行。最后输出结果可以选择用 `beam_search` 或 `greedy`，也是在`configs/model.json`里改。

### 数据获取和数据处理

我们只要获取到正确的latex公式就行。因为我们可以使用脚本将latex渲染出图片，所以就不用图片数据了。

原来我们想使用爬虫爬取[arXiv](https://arxiv.org/)的论文，然后通过正则表达式提取论文里的latex公式。

但是最后我们发现已经有人做了这个工作，所以就用了他们的公式数据。[im2latex-100k , arXiv:1609.04938](https://zenodo.org/record/56198#.XKMMU5gzZBB)

现在我们获取到latex公式数据，下面进行规范化。

> 为什么要规范化：如果不规范化，我们构建字典时就只能是char wise，而latex中有很多是有特定排列的指令，比如`\lim`，这样模型需要花费额外的神经元来记住这些pattern，会使模型效果变差，也导致训练费时间。（有时根本不收敛...别问我怎么知道的...）

我们先手动在代码编辑器里对数据进行规范化，很玄学地用了一些正则表达式，一步一步进行规范化。

最后总结了一下，明确要构建的字典大概是什么样的，然后写了脚本来处理。

然后是通过公式生成图片，保存公式-图片映射文件，构建字典。

构建字典很简单，遍历公式文件的每一行，然后以空格符` `为分隔符分割成若干latex块，去掉每一块首尾空格，若非空则加入字典集，保证不重复。

保存公式-图片映射文件也很简单，就是在渲染出图片后，保存`当前的公式在公式文件里的行号`和`图片路径`，写入映射文件里，也就是`.matching.txt`文件。图片文件名是直接用公式行号来命名的，比如`1234.png 1234`表示第1234行公式的公式图片是1234.png。所以知道了行号，就知道了公式图片路径。

通过公式生成图片稍微复杂一点，需要用到几个库：latex、ghostscript和magick。事实上用Katex也是可以的，katex是一个渲染latex公式的js库，体积小速度快。原来我们也是打算用这个库处理，后来因为环境问题放弃了。

latex原先我的环境里有了，这是用来生成pdf文件的。执行脚本后会得到A4纸大小的一页pdf。

ghostscript和magick绑定在一起，用来把pdf转化为png格式的图片。

转化为图片后，选定公式 padding 8个像素的方框，crop框外的空白，然后灰度化。

### 模型构建

让我鸽一段时间。。。有空再写！


## 6. 踩坑记录

### win10 用 GPU 加速训练

装驱动后就行了。运行下面代码进行训练：

```
C:/Users/dlink/AppData/Local/Programs/Python/Python35/python.exe train.py --data=configs/data.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/full/
```

我的环境比较奇葩，是win10+ubuntu wsl，也就是windows+linux子系统。我的GPU驱动装在win里面了，linux里没装，不过文件系统是共用的，所以用win的GPU驱动来训练模型。（嗯，python也有两套版本哈哈哈，都是python3.5）

### 如何可视化 Attention 层

在Attention层内自定义一个op，通过这个op把Attention传递到一个全局变量里。其他程序在模型预测完公式后，就可以在这个全局变量里获取到Attention。

## 致谢

十分感谢 Harvard 和 Guillaume Genthial 、Kelvin Xu 等人提供巨人的肩膀。

论文：
1. [Show, Attend and Tell(Kelvin Xu...)](https://arxiv.org/abs/1502.03044)
2. [Harvard's paper and dataset](http://lstm.seas.harvard.edu/latex/)
3. [Seq2Seq for LaTeX generation](https://guillaumegenthial.github.io/image-to-latex.html).