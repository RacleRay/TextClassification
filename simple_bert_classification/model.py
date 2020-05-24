# -*- coding:utf-8 -*-
# author: Racle
# project: simple_sentiment_classification

import os
from argparse import Namespace

import torch
from loguru import logger
from transformers import BertConfig, BertForSequenceClassification
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
# from tensorboardX import SummaryWriter
from dataloader import MyDataset


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

arg = {
    "batch_size": 8,
    "bert_model_dir": ".\chinese_wwm_pytorch",
    "num_labels": 10,
    "train_file": 'cnews\cnews.train.txt',
    "eval_file": 'cnews\cnews.val.txt',
    "test_file": None,
    "lr": 2e-5,
    "epochs": 10,
    "save_model":False,
    }


def collate_fn(samples):
    tokens_tensors = [s[0] for s in samples]
    if samples[0][1] is not None:  # [0]: batch dim
        labels = torch.stack([s[1] for s in samples])
    else:
        labels = None
    # padding
    tokens_tensors = pad_sequence(tokens_tensors,  batch_first=True)
    return tokens_tensors, labels


def train_one_epoch(model, loss_fn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            collate_fn=collate_fn)
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()

        loss, logits = model(batch, labels=labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred_labels = torch.argmax(logits, axis=1)
        train_acc += (pred_labels == labels).sum().item()
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def evaluate_one_epoch(model, loss_fn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model.eval()

    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)[0]

            error = loss_fn(logits, labels)
            loss += error.item()

            pred_labels = torch.argmax(logits, axis=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)

    return loss, acc


def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def train(bert, epochs, batch_size, save=False):
    trainset = MyDataset(arg['train_file'])
    devset = MyDataset(arg['eval_file'])
    if arg['test_file']:
        testset = MyDataset(arg['test_file'])

    config = BertConfig.from_pretrained(os.path.join(bert, 'bert_config.json'))
    config.num_labels = 20

    model = BertForSequenceClassification.from_pretrained(os.path.join(bert, 'pytorch_model.bin'),
                                                          config=config)
    for name, module in model.named_children():
        if name == 'bert':
            for param in module.parameters():
                param.requires_grad = False
        else:
            num_params = {sum(p.numel() for p in get_learnable_params(module))}
            print(f"可学习参数量为：{num_params}")

    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg['lr'])

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loss_fn, optimizer, trainset, batch_size=batch_size)
        val_loss, val_acc = evaluate_one_epoch(model, loss_fn, optimizer, devset, batch_size=batch_size)
        logger.info(f"epoch={epoch}")
        logger.info(f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}")
        logger.info(f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}")
        if save:
            label = "fine"
            torch.save(model, f"{bert}__{label}__e{epoch}.pickle")
    logger.success("Done!")


if __name__ == "__main__":
    train(arg['bert_model_dir'], arg['epochs'], arg['batch_size'], arg['save_model'])