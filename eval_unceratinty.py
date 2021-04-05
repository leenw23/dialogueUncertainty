"""
1. Single-candidate prediction을 할 때 MCDroput의 variance
2. Multi-candidates prediction을 할 때의 entropy
를 통해 uncertainty를 measure하는 스크립트 파일.
"""
import argparse
import json
import os

# pip install uncertainty-calibration
import calibration as cal
import numpy as np
import scipy
import torch
import torch.nn as nn
import transformers
from scipy.stats import entropy
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

from preprocess_dataset import get_dd_corpus, get_dd_multiref_testset
from utils import (RankerDataset, get_uttr_token, load_model,
                   make_random_negative_for_multi_ref, set_random_seed)


def modeling_mcdropout_uncertainty(
    model, ids, masks, softmax_op, num_forward_pass: int = 10
):
    """한 sample에 대한 예측 및 uncertainty value를 반환

    Args:
        model Deterministic BERT ranker
        ids (torch.Tensor([1,seq_len]))
        masks (torch.Tensor([1,seq_len]))
        num_forward_pass (int, optional): 몇번 forward pass해서 분산을 구할건지. Defaults to 10.
    """
    model.train()
    prediction_list = []
    for forward_pass in range(num_forward_pass):
        with torch.no_grad():
            prediction_list.append(
                float(softmax_op(model(ids, masks)[0]).cpu().numpy()[0][1])
            )
    return np.mean(prediction_list), np.var(prediction_list)


def mcdrop_main(args):
    set_random_seed(42)
    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    UTTR_TOKEN = get_uttr_token()
    special_tokens_dict = {"additional_special_tokens": [UTTR_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    model = load_model(model, args.model_path, 0, len(tokenizer))
    model.to(device)

    raw_dd_test = get_dd_corpus("test")
    test_dataset = RankerDataset(raw_dd_test, tokenizer, "test", 128, UTTR_TOKEN)
    softmax = torch.nn.Softmax(dim=1)

    prediction_list, uncertainty_list, label_list = [], [], []
    for idx in tqdm(range(len(test_dataset))):
        with torch.no_grad():
            sample = [el[idx] for el in test_dataset.feature]
            ids, masks = [torch.unsqueeze(el, 0).to(device) for el in sample[:2]]
            prediction, uncertainty = modeling_mcdropout_uncertainty(
                model,
                ids,
                masks,
                softmax,
            )
            prediction_list.append(prediction)
            uncertainty_list.append(uncertainty)
            # 1 for negative and 0 for random
            label_list.append(int(sample[2].numpy()))

        assert len(prediction_list) == len(label_list) == len(uncertainty_list)
    discrete_prediction_list = [1 if el >= 0.5 else 0 for el in prediction_list]
    accuracy = accuracy_score(label_list, discrete_prediction_list)
    f1 = f1_score(label_list, discrete_prediction_list)
    calibration_error = cal.get_ece(prediction_list, label_list)
    print("Accuracy: {}".format(accuracy))
    print("f1: {}".format(f1))
    print("ECE: {}".format(calibration_error))


def multi_candidate_main(args):
    set_random_seed(42)
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
    multiref_dd_dataset = get_dd_multiref_testset()
    multiref_dd_dataset_with_random_negative = make_random_negative_for_multi_ref(
        multiref_dd_dataset
    )

    prediction_list = []
    for sample_indx, sample in enumerate(tqdm(multiref_dd_dataset)):
        context, positive_responses, negative_responses = sample
        assert isinstance(context, str)
        assert isinstance(positive_responses, list) and all(
            [isinstance(el, str) for el in positive_responses]
        )
        assert isinstance(negative_responses, list) and all(
            [isinstance(el, str) for el in negative_responses]
        )
        softmax = torch.nn.Softmax(dim=1)
        positive_prediction = []
        negative_prediction = []

        for response in positive_responses:
            input_ids = tokenizer(
                context,
                text_pair=response,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            with torch.no_grad():
                input_ids = input_ids.to(device)

                positive_prediction.append(
                    (float(softmax(model(input_ids)[0]).cpu().numpy()[0][1]))
                )
        for response in negative_responses:
            input_ids = tokenizer(
                context,
                text_pair=response,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            with torch.no_grad():
                input_ids = input_ids.to(device)
                negative_prediction.append(
                    (float(softmax(model(input_ids)[0]).cpu().numpy()[0][1]))
                )
        # assert len(positive_prediction) == len(negative_prediction)
        prediction_list.append([positive_prediction, negative_prediction])

    with open("pos_neg_pred_scores.json", "w") as f:
        for item in prediction_list:
            json.dump({"positive": item[0], "negative": item[1]}, f)
            f.write("\n")


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
    parser.add_argument("--num_forward_pass", type=int, default=10)
    # mcdrop_main(args)
    multi_candidate_main(args)
