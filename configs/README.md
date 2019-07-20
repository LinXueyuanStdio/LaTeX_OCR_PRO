# 配置文件说明

## 数据
```json
{
    "export_name": "data.json",

    "dir_images_train": "data/images_train/", // 图片文件夹
    "dir_images_test" : "data/images_test/",
    "dir_images_val"  : "data/images_val/",

    "path_matching_train": "data/train.matching.txt", // 映射文件
    "path_matching_val"  : "data/val.matching.txt",
    "path_matching_test" : "data/test.matching.txt",

    "path_formulas_train": "data/train.formulas.norm.txt", // 公式文件
    "path_formulas_test" : "data/test.formulas.norm.txt",
    "path_formulas_val"  : "data/val.formulas.norm.txt",

    "bucket_train": true, // 分批次
    "bucket_val": true, // 分批次
    "bucket_test": true, // 分批次

    "max_iter"          : null, // 最大迭代次数
    "max_length_formula": 150, // 最大公式长度

    "buckets": [
        [240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100],
        [560, 80], [560, 100], [640, 80], [640, 100], [720, 80], [720, 100],
        [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
        [1000, 400], [1200, 200], [1600, 200], [1600, 1600]
        ] // 只支持以上大小的图片输入，如果输入的图片不是以上大小之一，就需要先 padding 成以上大小
}
```
## 模型
```json
{
    "export_name": "model.json",

    "model_name": "img2seq", // 模型名字

    "encoder_cnn": "vanilla", // 编码器

    "positional_embeddings": true, // 位置嵌入，不然注意力机制注意不到位置信息，只注意到先后信息
    "attn_cell_config": { // 注意力机制
        "cell_type": "lstm",
        "num_units": 512,
        "dim_e": 256,
        "dim_o": 512,
        "dim_embeddings": 80
    },

    "decoding": "beam_search", // 解码器
    "beam_size": 2,
    "div_gamma": 1,
    "div_prob": 0,
    "max_length_formula": 150 // 最大公式长度
}
```
## 训练过程（超参数）
```json
{
    "export_name": "training.json",

    "device": "cuda:3", // 多 GPU
    "criterion_method": "CrossEntropyLoss", // 损失

    "n_epochs": 40, // 训练次数
    "batch_size": 3, // 批次
    "dropout": 1, // dropout
    "clip": -1, // 截断

    "lr_scheduler":"CosineAnnealingLR", // 动态学习率
    "lr_method": "Adam", // 优化器
    "lr_init": 1e-3, // 学习率初始值
    "lr_min": 1e-4, // 学习率最小值
    "start_decay": 6,
    "end_decay": 13,
    "lr_warm": 1e-4,
    "end_warm": 2
}
```
## 字典
```json
{
	"export_name": "vocab.json",

    "unk": "_UNK", // unknown
    "pad": "_PAD", // padding
    "end": "_END", // end
    "path_vocab": "data/vocab.txt", // 字典的保存路径
    "min_count_tok": 10
}
```