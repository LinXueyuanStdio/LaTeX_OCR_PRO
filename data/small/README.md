```shell
data/small/
├── formulas
│   ├── train.formulas.norm.txt 规范化后的公式，以空格为分隔符直接构造字典
│   ├── test.formulas.norm.txt
│   ├── val.formulas.norm.txt
│   └── vocab.txt
├── images
│   ├── images_train 印刷体
│   ├── images_test
│   └── images_val
├── matching
│   ├── train.matching.txt 样式为 <image.png>, <formulas_id> 的匹配文件
│   ├── test.matching.txt
│   └── val.matching.txt
├── data.json
└── README.md
```