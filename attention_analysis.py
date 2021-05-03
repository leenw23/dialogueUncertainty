import argparse
import os

# pip install uncertainty-calibration
import calibration as cal
import numpy as np
import scipy
import torch
import torch.nn as nn
import transformers
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForNextSentencePrediction,
    BertModel,
    BertTokenizer,
)

from preprocess_dataset import get_dd_corpus
from utils import (
    RankerDataset,
    dump_config,
    get_uttr_token,
    load_model,
    set_random_seed,
    write2tensorboard,
)


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
    STEP1. Simple evaluation using Accracy and F1 on DD trainset.
    """
    raw_dd_train = get_dd_corpus("train")
    train_dataset = RankerDataset(raw_dd_train, tokenizer, "train", 128, UTTR_TOKEN)
    softmax = torch.nn.Softmax(dim=1)

    saver = []
    for idx in tqdm(range(len(train_dataset))):
        with torch.no_grad():
            sample = [el[idx] for el in train_dataset.feature]
            ids, masks = [torch.unsqueeze(el, 0).to(device) for el in sample[:2]]
            # 1 for positive and 0 for random
            if int(sample[2].numpy()) != 1:

                continue

            output = model(ids, masks, output_attentions=True, return_dict=True)
            attention_output = [el.cpu().numpy() for el in output["attentions"]]
            attention_output = sum([sum(sum(el[0])) for el in attention_output])
            attention_output = [el for el in attention_output if el != 0]
            prediction = softmax(output["logits"]).cpu().numpy()[0]
        assert len(prediction) == 2
        final_item = {
            "prediction": prediction[1],
            "attention": attention_output,
            "feature": [int(el) for el in train_dataset.feature[0][idx].numpy() if el != 0],
        }
        saver.append(final_item)
        # draw_and_save(tokenizer, final_item, args.attention_img_fname.format(idx))

    import pickle

    with open(args.attention_dump_fname, "wb") as f:
        pickle.dump(saver, f)


def draw_and_save(tokenizer, item, save_fname):
    os.makedirs(os.path.dirname(save_fname), exist_ok=True)
    ids = item["feature"]
    attention = item["attention"]
    if ids.count(102) != 2:
        return 1
    context = ids[: ids.index(102)][1:]
    context = tokenizer.convert_ids_to_tokens(context)
    response = ids[2 + len(context) :][:-1]
    response = tokenizer.decode(response)
    attention = attention[1 : 1 + len(context)]
    for idx, tok in enumerate(context):
        if tok in [".", ",", "?", "!", "[UTTR]"]:
            attention[idx] = 0
    attention /= sum(attention)
    fig, ax = plt.subplots(figsize=(len(context), 5))
    im = ax.imshow([attention])
    ax.set_xticks(np.arange(len(context)))
    ax.set_xticklabels(context)
    for score_idx, score in enumerate(attention):
        ax.text(
            score_idx,
            0,
            str(round(score, 3))[1:],
            ha="center",
            va="center",
            color="red",
            fontsize=12,
        )
    ax.text(0, 2, "Response: " + response, color="black", fontsize=12)
    ax.text(
        0,
        3,
        "Prediction: " + str(round(item["prediction"], 2))[1:],
        color="black",
        fontsize=12,
    )
    plt.savefig(save_fname)
    plt.close("all")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="deterministic_bert_ranker",
    )
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--attention_dump_fname", type=str, default="attention.pck")
    parser.add_argument("--attention_img_fname", type=str, default="img/attn_{}.png")

    args = parser.parse_args()
    args.exp_path = os.path.join(args.log_path, args.exp_name)
    args.model_path = os.path.join(args.exp_path, "model")
    args.board_path = os.path.join(args.exp_path, "board")
    main(args)
