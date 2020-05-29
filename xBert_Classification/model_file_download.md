https://huggingface.co/models?search=chinese  



# CLUE Pretrained Models

https://github.com/CLUEbenchmark/CLUEPretrainedModels

| 模型简称                               | 参数量 | 存储大小 | 语料               | 词汇表        | 直接下载                                                     |
| -------------------------------------- | ------ | -------- | ------------------ | ------------- | ------------------------------------------------------------ |
| **`RoBERTa-tiny-clue`** 超小模型       | 7.5M   | 28.3M    | **CLUECorpus2020** | **CLUEVocab** | **[TensorFlow](https://storage.googleapis.com/cluebenchmark/pretrained_models/RoBERTa-tiny-clue.zip) [PyTorch (提取码:8qvb)](https://pan.baidu.com/s/1hoR01GbhcmnDhZxVodeO4w)** |
| **`RoBERTa-tiny-pair`** 超小句子对模型 | 7.5M   | 28.3M    | **CLUECorpus2020** | **CLUEVocab** | **[TensorFlow](https://storage.googleapis.com/cluebenchmark/pretrained_models/RoBERTa-tiny-pair.zip) [PyTorch (提取码:8qvb)](https://pan.baidu.com/s/1hoR01GbhcmnDhZxVodeO4w)** |
| **`RoBERTa-tiny3L768-clue`** 小模型    | 38M    | 110M     | **CLUECorpus2020** | **CLUEVocab** | **[TensorFlow](https://storage.googleapis.com/cluebenchmark/pretrained_models/RoBERTa-tiny3L768-clue.zip)** |
| **`RoBERTa-tiny3L312-clue`** 小模型    | <7.5M  | 24M      | **CLUECorpus2020** | **CLUEVocab** | **[TensorFlow](https://storage.googleapis.com/cluebenchmark/pretrained_models/RoBERTa-tiny3L312-clue.zip)** |
| **`RoBERTa-large-clue`** 大模型        | 290M   | 1.20G    | **CLUECorpus2020** | **CLUEVocab** | **[TensorFlow](https://storage.googleapis.com/cluebenchmark/pretrained_models/RoBERTa-large-clue.zip) [PyTorch (提取码:8qvb)](https://pan.baidu.com/s/1hoR01GbhcmnDhZxVodeO4w)** |
| **`RoBERTa-large-pair`** 大句子对模型  | 290M   | 1.20G    | **CLUECorpus2020** | **CLUEVocab** | **[TensorFlow](https://storage.googleapis.com/cluebenchmark/pretrained_models/RoBERTa-large-pair.zip) [PyTorch (提取码:8qvb)](https://pan.baidu.com/s/1hoR01GbhcmnDhZxVodeO4w)** |

### 快速加载

依托于[Huggingface-Transformers 2.5.1](https://github.com/huggingface/transformers)，可轻松调用以上模型。

```
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```

其中`MODEL_NAME`对应列表如下：

| 模型名                     | MODEL_NAME                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **RoBERTa-tiny-clue**      | [`clue/roberta_chinese_clue_tiny`](https://huggingface.co/clue/roberta_chinese_clue_tiny) |
| **RoBERTa-tiny-pair**      | [`clue/roberta_chinese_pair_tiny`](https://huggingface.co/clue/roberta_chinese_pair_tiny) |
| **RoBERTa-tiny3L768-clue** | [`clue/roberta_chinese_3L768_clue_tiny`](https://huggingface.co/clue/roberta_chinese_3L768_clue_tiny) |
| **RoBERTa-tiny3L312-clue** | [`clue/roberta_chinese_3L312_clue_tiny`](https://huggingface.co/clue/roberta_chinese_3L312_clue_tiny) |
| **RoBERTa-large-clue**     | [`clue/roberta_chinese_clue_large`](https://huggingface.co/clue/roberta_chinese_clue_large) |
| **RoBERTa-large-pair**     | [`clue/roberta_chinese_pair_large`](https://huggingface.co/clue/roberta_chinese_pair_large) |



# RoBERTa

https://github.com/brightmart/roberta_zh

### 中文预训练RoBERTa模型-下载

*** 6层RoBERTa体验版 *** RoBERTa-zh-Layer6: [Google Drive](https://drive.google.com/file/d/1QXFqD6Qm8H9bRSbw7yZIgTGxD0O6ejUq/view?usp=sharing) 或 [百度网盘](https://pan.baidu.com/s/1TfKz-d9wvfqct8vN0c-vjg)，TensorFlow版本，Bert 直接加载, 大小为200M

** 推荐 RoBERTa-zh-Large 通过验证**

RoBERTa-zh-Large: [Google Drive ](https://drive.google.com/open?id=1W3WgPJWGVKlU9wpUYsdZuurAIFKvrl_Y)或 [百度网盘](https://pan.baidu.com/s/1Rk_QWqd7-wBTwycr91bmug) ，TensorFlow版本，Bert 直接加载

RoBERTa-zh-Large: [Google Drive ](https://drive.google.com/open?id=1yK_P8VhWZtdgzaG0gJ3zUGOKWODitKXZ)或 [百度网盘](https://pan.baidu.com/s/1MRDuVqUROMdSKr6HD9x1mw) ，PyTorch版本，Bert的PyTorch版直接加载

RoBERTa 24/12层版训练数据：30G原始文本，近3亿个句子，100亿个中文字(token)，产生了2.5亿个训练数据(instance)；

覆盖新闻、社区问答、多个百科数据等；

本项目与中文预训练24层XLNet模型 [XLNet_zh](https://github.com/brightmart/xlnet_zh)项目，使用相同的训练数据。

RoBERTa_zh_L12: [Google Drive](https://drive.google.com/open?id=1ykENKV7dIFAqRRQbZIh0mSb7Vjc2MeFA) 或 [百度网盘](https://pan.baidu.com/s/1hAs7-VSn5HZWxBHQMHKkrg) TensorFlow版本，Bert 直接加载

RoBERTa_zh_L12: [Google Drive](https://drive.google.com/open?id=1H6f4tYlGXgug1DdhYzQVBuwIGAkAflwB) 或[百度网盘](https://pan.baidu.com/s/1AGC76N7pZOzWuo8ua1AZfw) PyTorch版本，Bert的PyTorch版直接加载

------

[Roberta_l24_zh_base](https://drive.google.com/file/d/1cg3tVKPyUEmiI88H3gasqYC4LV4X8dNm/view?usp=sharing) TensorFlow版本，Bert 直接加载

24层base版训练数据：10G文本，包含新闻、社区问答、多个百科数据等。



# albert_zh

https://github.com/brightmart/albert_zh

【使用场景】任务相对比较简单一些或实时性要求高的任务，如语义相似度等句子对任务、分类任务；比较难的任务如阅读理解等，可以使用其他大模型。

1、[albert_tiny_zh](https://storage.googleapis.com/albert_zh/albert_tiny.zip), [albert_tiny_zh(训练更久，累积学习20亿个样本)](https://storage.googleapis.com/albert_zh/albert_tiny_489k.zip)，文件大小16M、参数为4M

```
训练和推理预测速度提升约10倍，精度基本保留，模型大小为bert的1/25；语义相似度数据集LCQMC测试集上达到85.4%，相比bert_base仅下降1.5个点。

训练使用如下参数： --max_seq_length=128 --train_batch_size=64   --learning_rate=1e-4   --num_train_epochs=5 

albert_tiny使用同样的大规模中文语料数据，层数仅为4层、hidden size等向量维度大幅减少; 尝试使用如下学习率来获得更好效果：{2e-5, 6e-5, 1e-4} 
```

1.1、[albert_tiny_google_zh(累积学习10亿个样本,google版本)](https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip)，模型大小16M、性能与albert_tiny_zh一致

1.2、[albert_small_google_zh(累积学习10亿个样本,google版本)](https://storage.googleapis.com/albert_zh/albert_small_zh_google.zip)，

```
 速度比bert_base快4倍；LCQMC测试集上比Bert下降仅0.9个点；去掉adam后模型大小18.5M；使用方法，见 #下游任务 Fine-tuning on Downstream Task     
```

2、[albert_large_zh](https://storage.googleapis.com/albert_zh/albert_large_zh.zip),参数量，层数24，文件大小为64M

```
参数量和模型大小为bert_base的六分之一；在口语化描述相似性数据集LCQMC的测试集上相比bert_base上升0.2个点
```

3、[albert_base_zh(额外训练了1.5亿个实例即 36k steps * batch_size 4096)](https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip); [albert_base_zh(小模型体验版)](https://storage.googleapis.com/albert_zh/albert_base_zh.zip), 参数量12M, 层数12，大小为40M

```
参数量为bert_base的十分之一，模型大小也十分之一；在口语化描述相似性数据集LCQMC的测试集上相比bert_base下降约0.6~1个点；
相比未预训练，albert_base提升14个点
```

4、[albert_xlarge_zh_177k ](https://storage.googleapis.com/albert_zh/albert_xlarge_zh_177k.zip); [albert_xlarge_zh_183k(优先尝试)](https://storage.googleapis.com/albert_zh/albert_xlarge_zh_183k.zip)参数量，层数24，文件大小为230M

```
参数量和模型大小为bert_base的二分之一；需要一张大的显卡；完整测试对比将后续添加；batch_size不能太小，否则可能影响精度
```

### 快速加载

依托于[Huggingface-Transformers 2.2.2](https://github.com/huggingface/transformers)，可轻松调用以上模型。

```
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")
model = AutoModel.from_pretrained("MODEL_NAME")
```

其中`MODEL_NAME`对应列表如下：

| 模型名                          | MODEL_NAME                     |
| ------------------------------- | ------------------------------ |
| albert_tiny_google_zh           | voidful/albert_chinese_tiny    |
| albert_small_google_zh          | voidful/albert_chinese_small   |
| albert_base_zh (from google)    | voidful/albert_chinese_base    |
| albert_large_zh (from google)   | voidful/albert_chinese_large   |
| albert_xlarge_zh (from google)  | voidful/albert_chinese_xlarge  |
| albert_xxlarge_zh (from google) | voidful/albert_chinese_xxlarge |





# BERT-wwm

https://github.com/ymcui/Chinese-BERT-wwm

本目录中主要包含base模型，故我们不在模型简称中标注`base`字样。对于其他大小的模型会标注对应的标记（例如large）。

- **`BERT-large模型`**：24-layer, 1024-hidden, 16-heads, 330M parameters
- **`BERT-base模型`**：12-layer, 768-hidden, 12-heads, 110M parameters

| 模型简称                                | 语料                      | Google下载                                                   | 讯飞云下载                                                   |
| --------------------------------------- | ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **`RBTL3, Chinese`**                    | **中文维基+ 通用数据[1]** | **[TensorFlow](https://drive.google.com/open?id=1Jzn1hYwmv0kXkfTeIvNT61Rn1IbRc-o8)** **[PyTorch](https://drive.google.com/open?id=1qs5OasLXXjOnR2XuGUh12NanUl0pkjEv)** | **[TensorFlow（密码vySW）](https://pan.iflytek.com/link/0DD18FAC080BAF75DBA28FB5C0047760)** **[PyTorch（密码rgCs）](https://pan.iflytek.com/link/7C6A513BED2D42170B6DBEE5A866FB3F)** |
| **`RBT3, Chinese`**                     | **中文维基+ 通用数据[1]** | **[TensorFlow](https://drive.google.com/open?id=1-rvV0nBDvRCASbRz8M9Decc3_8Aw-2yi)** **[PyTorch](https://drive.google.com/open?id=1_LqmIxm8Nz1Abvlqb8QFZaxYo-TInOed)** | **[TensorFlow（密码b9nx）](https://pan.iflytek.com/link/275E5B46185C982D4AF5AC295E1651B6)** **[PyTorch（密码Yoep）](https://pan.iflytek.com/link/A094EB0A73B1E7209FEBC6C5CF7AEF27)** |
| **`RoBERTa-wwm-ext-large, Chinese`**    | **中文维基+ 通用数据[1]** | **[TensorFlow](https://drive.google.com/open?id=1dtad0FFzG11CBsawu8hvwwzU2R0FDI94)** **[PyTorch](https://drive.google.com/open?id=1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq)** | **[TensorFlow（密码u6gC）](https://pan.iflytek.com/link/AC056611607108F33A744A0F56D0F6BE)** **[PyTorch（密码43eH）](https://pan.iflytek.com/link/9B46A0ABA70C568AAAFCD004B9A2C773)** |
| **`RoBERTa-wwm-ext, Chinese`**          | **中文维基+ 通用数据[1]** | **[TensorFlow](https://drive.google.com/open?id=1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt)** **[PyTorch](https://drive.google.com/open?id=1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25)** | **[TensorFlow（密码Xe1p）](https://pan.iflytek.com/link/98D11FAAF0F0DBCB094EE19CCDBC98BF)** **[PyTorch（密码waV5）](https://pan.iflytek.com/link/92ADD2C34C91F3B44E0EC97F101F89D8)** |
| **`BERT-wwm-ext, Chinese`**             | **中文维基+ 通用数据[1]** | **[TensorFlow](https://drive.google.com/open?id=1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi)** **[PyTorch](https://drive.google.com/open?id=1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_)** | **[TensorFlow（密码4cMG）](https://pan.iflytek.com/link/653637473FFF242C3869D77026C9BDB5)** **[PyTorch（密码XHu4）](https://pan.iflytek.com/link/B9ACE1C9F228A0F42242672EF6CE1721)** |
| **`BERT-wwm, Chinese`**                 | **中文维基**              | **[TensorFlow](https://drive.google.com/open?id=1RoTQsXp2hkQ1gSRVylRIJfQxJUgkfJMW)** **[PyTorch](https://drive.google.com/open?id=1AQitrjbvCWc51SYiLN-cJq4e0WiNN4KY)** | **[TensorFlow（密码07Xj）](https://pan.iflytek.com/link/A2483AD206EF85FD91569B498A3C3879)** **[PyTorch（密码hteX）](https://pan.iflytek.com/link/5DBDD89414E5B565D3322D6B7937DF47)** |
| `BERT-base, Chinese`Google              | 中文维基                  | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) | -                                                            |
| `BERT-base, Multilingual Cased`Google   | 多语种维基                | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) | -                                                            |
| `BERT-base, Multilingual Uncased`Google | 多语种维基                | [Google Cloud](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) | -                                                            |

> [1] 通用数据包括：百科、新闻、问答等数据，总词数达5.4B，处理后的文本大小约10G

以上预训练模型以TensorFlow版本的权重为准。 对于PyTorch版本，我们使用的是由Huggingface出品的[PyTorch-Transformers 1.0](https://github.com/huggingface/pytorch-transformers)提供的转换脚本。 如果使用的是其他版本，请自行进行权重转换。 中国大陆境内建议使用讯飞云下载点，境外用户建议使用谷歌下载点，base模型文件大小约**400M**。 以TensorFlow版`BERT-wwm, Chinese`为例，下载完毕后对zip文件进行解压得到：

```
chinese_wwm_L-12_H-768_A-12.zip
    |- bert_model.ckpt      # 模型权重
    |- bert_model.meta      # 模型meta信息
    |- bert_model.index     # 模型index信息
    |- bert_config.json     # 模型参数
    |- vocab.txt            # 词表
```

其中`bert_config.json`和`vocab.txt`与谷歌原版`BERT-base, Chinese`完全一致。 PyTorch版本则包含`pytorch_model.bin`, `bert_config.json`, `vocab.txt`文件。

### 快速加载

#### 使用Huggingface-Transformers

依托于[Huggingface-Transformers 2.2.2](https://github.com/huggingface/transformers)，可轻松调用以上模型。

```
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```

**注意：本目录中的所有模型均使用BertTokenizer以及BertModel加载，请勿使用RobertaTokenizer/RobertaModel！**

其中`MODEL_NAME`对应列表如下：

| 模型名                | MODEL_NAME                        |
| --------------------- | --------------------------------- |
| RoBERTa-wwm-ext-large | hfl/chinese-roberta-wwm-ext-large |
| RoBERTa-wwm-ext       | hfl/chinese-roberta-wwm-ext       |
| BERT-wwm-ext          | hfl/chinese-bert-wwm-ext          |
| BERT-wwm              | hfl/chinese-bert-wwm              |
| RBT3                  | hfl/rbt3                          |
| RBTL3                 | hfl/rbtl3                         |



# ELECTRA

https://github.com/ymcui/Chinese-ELECTRA

本目录中包含以下模型，目前仅提供TensorFlow版本权重。

- **`ELECTRA-base, Chinese`**：12-layer, 768-hidden, 12-heads, 102M parameters
- **`ELECTRA-small, Chinese`**: 12-layer, 256-hidden, 4-heads, 12M parameters

| 模型简称                     | 语料              | Google下载                                                   | 讯飞云下载                                                   | 压缩包大小 |
| ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- |
| **`ELECTRA-base, Chinese`**  | 中文维基+通用数据 | [TensorFlow](https://drive.google.com/open?id=1FMwrs2weFST-iAuZH3umMa6YZVeIP8wD) [PyTorch-D](https://drive.google.com/open?id=1iBanmudRHLm3b4X4kL_FxccurDjL4RYe) [PyTorch-G](https://drive.google.com/open?id=1x-fcgS9GU8X51H1FFiqkh0RIDMGTTX7c) | [TensorFlow（密码3VQu）](https://pan.iflytek.com/link/43B111080BD4A2D3370423912B45491E) [PyTorch-D（密码WQ8r）](http://pan.iflytek.com/link/31F0C2FB919C6099DEC72FD72C0AFCFB) [PyTorch-G（密码XxnY）](http://pan.iflytek.com/link/2DD6237FE1B99ECD81F775FC2C272149) | 383M       |
| **`ELECTRA-small, Chinese`** | 中文维基+通用数据 | [TensorFlow](https://drive.google.com/open?id=1uab-9T1kR9HgD2NB0Kz1JB_TdSKgJIds) [PyTorch-D](https://drive.google.com/open?id=1A1wdw41kOFC3n3AjfFTRZHQdjCL84bsg) [PyTorch-G](https://drive.google.com/open?id=1FpdHG2UowDTIepiuOiJOChrtwJSMQJ6N) | [TensorFlow（密码wm2E）](https://pan.iflytek.com/link/E5B4E8FE8B22A5FF03184D34CB2F1767) [PyTorch-D（密码Cch4）](http://pan.iflytek.com/link/5AE514A3721E4E75A0E04B8E99BB4098) [PyTorch-G（密码xCH8）](http://pan.iflytek.com/link/CB800D74191E948E06B45238AB797933) | 46M        |

*PyTorch-D: discriminator, PyTorch-G: generator

中国大陆境内建议使用讯飞云下载点，境外用户建议使用谷歌下载点。 以TensorFlow版`ELECTRA-small, Chinese`为例，下载完毕后对zip文件进行解压得到：

```
chinese_electra_small_L-12_H-256_A-4.zip
    |- checkpoint                           # checkpoint信息
    |- electra_small.data-00000-of-00001    # 模型权重
    |- electra_small.meta                   # 模型meta信息
    |- electra_small.index                  # 模型index信息
    |- vocab.txt                            # 词表
```

### 训练细节

我们采用了大规模中文维基以及通用文本训练了ELECTRA模型，总token数达到5.4B，与[RoBERTa-wwm-ext系列模型](https://github.com/ymcui/Chinese-BERT-wwm)一致。词表方面沿用了谷歌原版BERT的WordPiece词表，包含21128个token。其他细节和超参数如下（未提及的参数保持默认）：

- `ELECTRA-base`: 12层，隐层768，12个注意力头，学习率2e-4，batch256，最大长度512，训练1M步
- `ELECTRA-small`: 12层，隐层256，4个注意力头，学习率5e-4，batch1024，最大长度512，训练1M步

### 快速加载

#### 使用Huggingface-Transformers

[Huggingface-Transformers 2.8.0](https://github.com/huggingface/transformers/releases/tag/v2.8.0)版本已正式支持ELECTRA模型，可通过如下命令调用。

```
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME) 
```

其中`MODEL_NAME`对应列表如下：

| 模型名                 | 组件          | MODEL_NAME                              |
| ---------------------- | ------------- | --------------------------------------- |
| ELECTRA-base, Chinese  | discriminator | hfl/chinese-electra-base-discriminator  |
| ELECTRA-base, Chinese  | generator     | hfl/chinese-electra-base-generator      |
| ELECTRA-small, Chinese | discriminator | hfl/chinese-electra-small-discriminator |
| ELECTRA-small, Chinese | generator     | hfl/chinese-electra-small-generator     |



# XLNet

https://github.com/ymcui/Chinese-XLNet

- **`XLNet-mid`**：24-layer, 768-hidden, 12-heads, 209M parameters
- **`XLNet-base`**：12-layer, 768-hidden, 12-heads, 117M parameters

| 模型简称                  | 语料                      | Google下载                                                   | 讯飞云下载                                                   |
| ------------------------- | ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **`XLNet-mid, Chinese`**  | **中文维基+ 通用数据[1]** | **[TensorFlow](https://drive.google.com/open?id=1342uBc7ZmQwV6Hm6eUIN_OnBSz1LcvfA)** **[PyTorch](https://drive.google.com/open?id=1u-UmsJGy5wkXgbNK4w9uRnC0RxHLXhxy)** | **[TensorFlow（密码Cpq8）](https://pan.iflytek.com/link/3DD1B2F248C5B33F3893829E9B7FCDA3)** **[PyTorch（密码VBE6）](https://pan.iflytek.com/link/DF1DACD696FAC4E0BEE4EB09674CA7D8)** |
| **`XLNet-base, Chinese`** | **中文维基+ 通用数据[1]** | **[TensorFlow](https://drive.google.com/open?id=1m9t-a4gKimbkP5rqGXXsEAEPhJSZ8tvx)** **[PyTorch](https://drive.google.com/open?id=1mPDgcMfpqAf2wk9Nl8OaMj654pYrWXaR)** | **[TensorFlow（密码DfNj）](https://pan.iflytek.com/link/AECE9CCD57DD58A498676FD71D0557F8)** **[PyTorch（密码6e3y）](https://pan.iflytek.com/link/AEF637509F3777F6526FF276AD19763C)** |

> [1] 通用数据包括：百科、新闻、问答等数据，总词数达5.4B，与我们发布的[BERT-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)训练语料相同。

以上预训练模型以TensorFlow版本的权重为准。 对于PyTorch版本，我们使用的是由Huggingface出品的[PyTorch-Transformers 1.0](https://github.com/huggingface/pytorch-transformers)提供的转换脚本。 如果使用的是其他版本，请自行进行权重转换。 中国大陆境内建议使用讯飞云下载点，境外用户建议使用谷歌下载点，`XLNet-mid`模型文件大小约**800M**。 以TensorFlow版`XLNet-mid, Chinese`为例，下载完毕后对zip文件进行解压得到：

```
chinese_xlnet_mid_L-24_H-768_A-12.zip
    |- xlnet_model.ckpt      # 模型权重
    |- xlnet_model.meta      # 模型meta信息
    |- xlnet_model.index     # 模型index信息
    |- xlnet_config.json     # 模型参数
    |- spiece.model          # 词表
```

### 快速加载

依托于[Huggingface-Transformers 2.2.2](https://github.com/huggingface/transformers)，可轻松调用以上模型。

```
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")
model = AutoModel.from_pretrained("MODEL_NAME")
```

其中`MODEL_NAME`对应列表如下：

| 模型名     | MODEL_NAME             |
| ---------- | ---------------------- |
| XLNet-mid  | hfl/chinese-xlnet-mid  |
| XLNet-base | hfl/chinese-xlnet-base |



### XLNet_zh_Large

https://github.com/brightmart/xlnet_zh

XLNet_zh_Large， [百度网盘](https://pan.baidu.com/s/1dy0Z27DoZdMpSmoz1Q4G5A)，或 [Google drive](https://github.com/brightmart/xlnet_zh#)，TensorFlow版本。暂时没有去掉adam参数，去掉后模型会变成1.3G左右。

```
XLNet_zh_Large_L-24_H-1024_A-16.zip 
  |- xlnet_model.ckpt    # 模型权重
  |- xlnet_model.index   # 模型meta信息
  |- xlnet_model.meta    # 模型index新
  |- xlnet_config.json： # 配置文件
  |- spiece.model:       # 词汇表
```

PyTorch版本，可使用类似的命名来转换，具体建[pytorch_transformers](https://github.com/brightmart/xlnet_zh#)项目：

```
python -u -m pytorch_transformers.convert_tf_checkpoint_to_pytorch --tf_checkpoint_path XLNet-zh-Large-PyTorch/ --bert_config_file XLNet-zh-Large-PyTorch/config.json --pytorch_dump_path XLNet-zh-Large-PyTorch/xlnet_zh_large_pytorch_model.bin
```



# adamlin/bert-distil-chinese

```
tokenizer = AutoTokenizer.from_pretrained("adamlin/bert-distil-chinese")

model = AutoModel.from_pretrained("adamlin/bert-distil-chinese")
```



# GPT2 for Multiple Languages

[gpt2-ml](https://github.com/imcaspar/gpt2-ml)

1.5B GPT2 pretrained Chinese model [**Google Drive\**](https://drive.google.com/file/d/1IzWpQ6I2IgfV7CldZvFJnZ9byNDZdO4n)

SHA256: 4a6e5124df8db7ac2bdd902e6191b807a6983a7f5d09fb10ce011f9a073b183e

Corpus from [THUCNews](http://thuctc.thunlp.org/#中文文本分类数据集THUCNews) and [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)

Using [Cloud TPU Pod v3-256](https://cloud.google.com/tpu/docs/types-zones#types) to train 10w steps



# 多领域开源中文预训练语言模型仓库

https://github.com/thunlp/OpenCLaP

### 模型概览

以下是我们目前公开发布的模型概览：

| 名称         | 基础模型  | 数据来源                            | 训练数据大小 | 词表大小 | 模型大小 | 下载地址                                                     |
| ------------ | --------- | ----------------------------------- | ------------ | -------- | -------- | ------------------------------------------------------------ |
| 民事文书BERT | bert-base | 全部民事文书                        | 2654万篇文书 | 22554    | 370MB    | [点我下载](https://thunlp.oss-cn-qingdao.aliyuncs.com/bert/ms.zip) |
| 刑事文书BERT | bert-base | 全部刑事文书                        | 663万篇文书  | 22554    | 370MB    | [点我下载](https://thunlp.oss-cn-qingdao.aliyuncs.com/bert/xs.zip) |
| 百度百科BERT | bert-base | [百度百科](http://baike.baidu.com/) | 903万篇词条  | 22166    | 367MB    | [点我下载](https://thunlp.oss-cn-qingdao.aliyuncs.com/bert/baike.zip) |

### 使用方式

我们提供的模型可以被开源项目[pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)直接使用。以民事文书BERT为例，具体使用方法分为两步：

- 首先使用脚本下载我们的模型

```
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/bert/ms.zip
unzip ms.zip
```

- 在运行时指定使用我们的模型`--bert_model $model_folder`来进行使用



# ZEN

https://github.com/sinovation/ZEN

A BERT-based Chinese Text Encoder Enhanced by N-gram Representations

