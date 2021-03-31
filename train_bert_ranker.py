import argparse
import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import transformers
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import (BertConfig, BertForNextSentencePrediction,
                          BertTokenizer)

from preprocess_dataset import get_dd_corpus
from utils import (RankerDataset, dump_config, get_uttr_token, load_model,
                   save_model, set_random_seed, write2tensorboard)


def main(args):
    set_random_seed(42)

    dump_config(args)
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    UTTR_TOKEN = get_uttr_token()
    special_tokens_dict = {"additional_special_tokens": [UTTR_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    raw_dd_train, raw_dd_dev = get_dd_corpus("train"), get_dd_corpus("validation")

    valid_dataset, train_dataset = (
        RankerDataset(raw_dd_train, tokenizer, "dev", 128, UTTR_TOKEN),
        RankerDataset(raw_dd_dev, tokenizer, "train", 128, UTTR_TOKEN),
    )

    trainloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
    )
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True)

    """
    Training
    """
    crossentropy = CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
    )
    writer = SummaryWriter(args.board_path)

    save_model(model, "begin", args.model_path)
    global_step = 0
    for epoch in range(args.epoch):
        print("Epoch {}".format(epoch))
        model.train()
        for step, batch in enumerate(tqdm(trainloader)):
            ids, masks, labels = tuple(el.to(device) for el in batch)
            output = model(ids, masks, next_sentence_label=labels)
            loss = output[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
            write2tensorboard(writer, {"loss": loss}, "train", global_step)
            global_step += 1

        model.eval()
        loss_list = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(validloader)):
                ids, masks, labels = tuple(el.to(device) for el in batch)
                output = model(ids, masks, next_sentence_label=labels)
                loss = output[0]
                loss_list.append(loss.cpu().detach().numpy())

                write2tensorboard(writer, {"loss": loss}, "train", global_step)
            final_loss = sum(loss_list) / len(loss_list)
            write2tensorboard(writer, {"loss": final_loss}, "valid", global_step)
        save_model(model, epoch, args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="deterministic_bert_ranker",
    )
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=3)

    args = parser.parse_args()

    args.exp_path = os.path.join(args.log_path, args.exp_name)
    args.model_path = os.path.join(args.exp_path, "model")
    args.board_path = os.path.join(args.exp_path, "board")
    os.makedirs(args.model_path, exist_ok=False)
    os.makedirs(args.board_path, exist_ok=False)
    main(args)
