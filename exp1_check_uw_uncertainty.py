"""
Unknown word로 바뀐 context를 사용했을 때 baseline들의 uncertainty가 얼마나 변화하는지.
"""

import argparse
import os

# pip install uncertainty-calibration
import calibration as cal
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
from transformers import BertConfig, BertForNextSentencePrediction, BertTokenizer, BertModel

from preprocess_dataset import get_dd_corpus, get_dd_multiref_testset
from utils import (
    recall_x_at_k,
    RankerDataset,
    make_uw_select_dataset,
    dump_config,
    get_uttr_token,
    load_model,
    get_nota_token,
    set_random_seed,
    write2tensorboard,
    SelectionDataset,
    get_uw_annotation,
)
import seaborn as sns


from selection_model import BertSelect
from eval_unceratinty import modeling_mcdropout_uncertainty
from matplotlib import pyplot as plt


def main():
    set_random_seed(42)

    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    UTTR_TOKEN = get_uttr_token()
    NOTA_TOKEN = get_nota_token()
    special_tokens_dict = {"additional_special_tokens": [UTTR_TOKEN, NOTA_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    bert = BertModel.from_pretrained("bert-base-uncased")
    bert.resize_token_embeddings(len(tokenizer))
    model = BertSelect(bert)
    model = load_model(model, "./logs/select_batch12_candi10_seed42/model", 0, len(tokenizer))
    model.to(device)

    softmax = torch.nn.Softmax(dim=1)

    uw_original_annotation = get_uw_annotation(
        "nw_annotation.txt", change_uw_to_original=True, replace_golden_to_nota=False, is_dev=False
    )
    uw_changed_annotation = get_uw_annotation(
        "nw_annotation.txt", change_uw_to_original=False, replace_golden_to_nota=False, is_dev=False
    )

    softmax = torch.nn.Softmax(dim=1)
    select_score_list, mcdrop_prediction_list = [], []

    for original, changed in tqdm(zip(uw_original_annotation, uw_changed_annotation)):
        org_ctx, org_rsp = original
        chd_ctx, chd_rsp = changed
        org_input = tokenizer([org_ctx], text_pair=[org_rsp], return_tensors="pt")
        org_ids, org_mask = org_input["input_ids"].to(device), org_input["attention_mask"].to(
            device
        )
        chd_input = tokenizer([chd_ctx], text_pair=[chd_rsp], return_tensors="pt")
        chd_ids, chd_mask = chd_input["input_ids"].to(device), chd_input["attention_mask"].to(
            device
        )
        with torch.no_grad():
            org_mcdrop_prediction, org_mcdrop_uncertainty = modeling_mcdropout_uncertainty(
                model, org_ids, org_mask
            )
            chd_mcdrop_prediction, chd_mcdrop_uncertainty = modeling_mcdropout_uncertainty(
                model, chd_ids, chd_mask
            )

            model.eval()
            org_select_prediction = float(model(org_ids, org_mask).cpu().numpy())
            chd_select_prediction = float(model(chd_ids, chd_mask).cpu().numpy())
        select_score_list.append([org_select_prediction, chd_select_prediction])
        mcdrop_prediction_list.append([org_mcdrop_prediction, chd_mcdrop_prediction])

    org_select_list = [el[0] for el in select_score_list]
    chd_select_list = [el[1] for el in select_score_list]
    org_mc_list = [el[0] for el in mcdrop_prediction_list]
    chd_mc_list = [el[1] for el in mcdrop_prediction_list]

    select_ratio = [(el[1] - el[0]) / el[0] for el in select_score_list]
    mcdrop_ratio = [(el[1] - el[0]) / el[0] for el in mcdrop_prediction_list]

    print("### Selection Model ###")
    print("1. Original: {}".format(round(sum(org_select_list) / len(org_select_list), 2)))
    print("2. UW-state: {}".format(round(sum(chd_select_list) / len(chd_select_list), 2)))
    print("3. Change ratio: {}".format(round(sum(select_ratio) / len(select_ratio), 2)))
    print("\n### MC-dropout Model ###")
    print("1. Original: {}".format(round(sum(org_mc_list) / len(org_mc_list), 2)))
    print("2. UW-state: {}".format(round(sum(chd_mc_list) / len(chd_mc_list), 2)))
    print("3. Change ratio: {}".format(round(sum(mcdrop_ratio) / len(mcdrop_ratio), 2)))

    fig, ax = plt.subplots()
    kwargs = dict(alpha=0.5, histtype="step")
    plt.hist(org_select_list, **kwargs, color="b", label="original")
    plt.hist(chd_select_list, **kwargs, color="r", label="changed")
    plt.title("BERT-select")
    plt.legend()
    plt.savefig("select.png")
    plt.cla()

    plt.hist(org_mc_list, **kwargs, color="b", label="original")
    plt.hist(chd_mc_list, **kwargs, color="r", label="changed")
    plt.title("MCdrop")
    plt.legend()
    plt.savefig("select_mcdrop.png")
    plt.cla()


if __name__ == "__main__":
    main()