import argparse
import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import accuracy_score, f1_score
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
                   set_random_seed, write2tensorboard)


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
    model = load_model(model, args.model_path, 0, len(tokenizer))
    model.to(device)

    """
    STEP1. Simple evaluation using Accracy and F1 on DD testset.
    """
    raw_dd_test = get_dd_corpus("test")
    test_dataset = RankerDataset(raw_dd_test, tokenizer, "test", 128, UTTR_TOKEN)
    softmax = torch.nn.Softmax(dim=1)

    prediction_list, label_list = [], []
    for idx in tqdm(range(len(test_dataset))):
        with torch.no_grad():
            sample = [el[idx] for el in test_dataset.feature]
            ids, masks = [torch.unsqueeze(el, 0).to(device) for el in sample[:2]]
            # 1 for positive and 0 for random
            label = int(sample[2].numpy())
            prediction = softmax(model(ids, masks)[0]).cpu().numpy()[0]
        assert len(prediction) == 2
        prediction = prediction[1]
        prediction_list.append(prediction)
    accuracy = accuracy_score(label_list, prediction_list)
    f1 = f1_score(label_list, prediction_list)
    print("Accuracy: {}".format(accuracy))
    print("f1: {}".format(f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="deterministic_bert_ranker",
    )
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--epoch", type=int, default=0)

    args = parser.parse_args()
    args.exp_path = os.path.join(args.log_path, args.exp_name)
    args.model_path = os.path.join(args.exp_path, "model")
    args.board_path = os.path.join(args.exp_path, "board")
    main(args)
