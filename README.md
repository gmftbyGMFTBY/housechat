# housechat
https://www.datafountain.cn/competitions/474

基于 BERT 预训练语言模型的问答对匹配建模

## 环境要求
* pytorch 1.5
* CUDA 9.2
* transformers 1.2.1

## 模型具体细节

在huggingface:transformers toolkit的基础上构建模型，因为数据集上是中文格式的，我们使用了官方释出的 `bert-base-chinese` 预训练参数。
本文中使用的方法类似于 `SA-BERT` 中的架构，因为问答匹配对是单轮并不是多轮的格式，所以我们在这里忽略了 `SA-BERT` 中的 speaker-aware 信息，
只利用了对应的utterance的 `token_type_ids` (segment embedding，使用segment embedding的原因在于辅助模型区分context和response) ，`position ids` 和 `mask` 信息进行训练，模型整体架构图如下：

![](./img/img1.png)

## 使用方法

1. train the model

```bash
./run.sh train bert 0,1,2,3
```

2. test the model
运行 test 之后，生成的文件存储在 `rest/bert/rest.txt` 中，可以直接提交

```bash
./run.sh test bert 0
```
