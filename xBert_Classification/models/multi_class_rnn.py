#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   multi_class_rnn.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    PreTrainedModel,
    BertModel,
    BertPreTrainedModel,
    AlbertModel,
    AlbertPreTrainedModel,
    XLNetModel,
    XLNetPreTrainedModel,
    DistilBertConfig,
    DistilBertModel,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    RobertaConfig,
    RobertaModel,
    ElectraConfig,
    ElectraModel,
    ElectraPreTrainedModel,
)
from transformers.modeling_roberta import RobertaClassificationHead, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_electra import ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_utils import SequenceSummary


"""
Bert类模型加上LSTM，结合multi_label_linear.py，可以轻松定义任意Bert类模型。

NOTE: 训练Bert类模型加上LSTM，很容易过拟合。因此，一般的方法是，一起fine-tuning到一个还不错的结果，不要求完全收敛。
然后，freezeBert类模型参数，再fine-tuning LSTM，得到最优的结果。LSTM一般采用双向。
"""

