
## 解决方案

### 在 win10 用 GPU 加速训练

1. 首先下载对应 `tensorflow-gpu==1.12.2` 版本的 `cuda==9.0` 和 `cudnn==7.0`
2. 安装 cuda。注意选择自定义安装 > 只勾选 lib 库，其他都不要。驱动可能比它要求的新，不要让它给你装驱动，会坏
3. 移动 cudnn 到 cuda 对应的路径下
4. 加环境变量
5. 最后就行啦

运行下面代码进行训练：

```
C:/Users/dlink/AppData/Local/Programs/Python/Python35/python.exe train.py --data=configs/data.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/full/
```

我的环境比较奇葩，是win10+ubuntu wsl，也就是windows+linux子系统。我的GPU驱动装在win里面了，linux里没装，不过文件系统是共用的，所以用win的GPU驱动来训练模型。（嗯，python也有两套版本哈哈哈，都是python3.5）

### 如何可视化 Attention 层

在Attention层内自定义一个op，通过这个op把Attention传递到一个全局变量里。其他程序在模型预测完公式后，就可以在这个全局变量里获取到Attention。
