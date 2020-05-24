#-*- coding:utf-8 -*-
# author: Racle
# project: simple_sentiment_classification

import glob
import re
import sys

import pandas as pd
import torch
from loguru import logger
from transformers import BertTokenizer
from torch.utils.data import Dataset


logger.info("Loading the tokenizer")
tokenizer = BertTokenizer.from_pretrained("./chinese_wwm_pytorch")

# import pytreebank
#
# 斯坦福的Stanford Sentiment Treebank数据集。有句子级别的情感分类，还有词级别、短语级别的分类。
# logger.info("Loading SST")
# sst = pytreebank.load_sst()


class MyDataset(Dataset):
    """Configurable Dataset.
    """
    def __init__(self, data_path, max_length=400):
        """Initializes the dataset with given configuration.
        """
        self.max_length = max_length
        self.data = pd.read_csv(data_path, delimiter="\t", names=['label', 'content'])

        # 过滤长度较长的句子
        # self.data = self.data[~(self.data.content.apply(lambda x : len(x)) > max_length)]
        # self.data.reindex()
        # clean data
        self.data['content'] = self.data.content.apply(self.clean_stentence)
        self.data['token_ids'] = self.data.content.apply(self.tokenize)  # encoded data

        self.label_map = {v: k for k, v in enumerate(self.data.label.unique())}
        self.inverse_label = {v: k for k, v in self.label_map.items()}
        self.data['label'] = self.data.label.apply(lambda x: self.label_map[x])

        self.weights = self.get_count_weight()
        print('Finished dataset initalization.')

    def tokenize(self, content):
        current_length = len(content)
        if current_length > self.max_length:
            encodes = tokenizer.encode(content[:self.max_length-1])
            return encodes
        else:
            encodes = tokenizer.encode(content)
            return encodes

    def get_count_weight(self):
        """数据集样本分布"""
        length = len(self)
        weights = {k: v / length for k, v in self.data.label.value_counts().items()}
        return weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sent = torch.tensor(self.data.token_ids[index])
        label = torch.tensor(self.data.label[index])
        return sent, label

    @staticmethod
    def clean_stentence(sent):
        sent = sent.replace('\n', '').replace('\t', '').replace('\u3000', '')
        # 删除对断句没有帮助的符号
        pattern_1 = re.compile(r"\(|\)|（|）|\"|“|”|\*|《|》|<|>|&|#|~|·|`|=|\+|\}|\{|\||、|｛|｝|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〿|–—|…|‧|﹏|")
        sent = re.sub(pattern_1, "", sent)
        # 断句符号统一为中文符号
        sent = re.sub(r"!", "！", sent)
        sent = re.sub(r"\?", "？", sent)
        sent = re.sub(r";", "；", sent)
        sent = re.sub(r",", "，", sent)
        # 去除网站，图片引用
        sent = re.sub(r"[！a-zA-z]+://[^\s]*", "", sent)
        # 去除邮箱地址
        sent = re.sub(r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*", "", sent)

        sent = re.sub(r"@", "", sent)
        sent = sent.replace(' ', '').lower()
        return sent