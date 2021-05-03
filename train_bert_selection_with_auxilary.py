import argparse
import os

import pickle
import numpy as np
import scipy
import torch
import torch.nn as nn
import transformers
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from selection_model import BertSelectAuxilary, BertSelect
from preprocess_dataset import get_dd_corpus
from utils import (
    corrupt_context_wordlevel_for_auxilary,
    RankerDataset,
    str2bool,
    dump_config,
    get_uttr_token,
    load_model,
    save_model,
    set_random_seed,
    SelectionDataset,
    write2tensorboard,
)


def main(args):
    set_random_seed(args.random_seed)

    dump_config(args)
    device = torch.device("cuda")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    UTTR_TOKEN = get_uttr_token()
    special_tokens_dict = {"additional_special_tokens": [UTTR_TOKEN, "[NOTA]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    tokens_to_skip = tokenizer.convert_tokens_to_ids(
        [".", ",", "?", "!", "[UTTR]", "[CLS]", "[SEP]"]
    )
    if not args.random_initialization:
        bert = BertModel.from_pretrained("bert-base-uncased")
    else:
        bert = BertModel(BertConfig())
    bert.resize_token_embeddings(len(tokenizer))
    model = BertSelectAuxilary(bert)
    model = torch.nn.DataParallel(model)
    model.to(device)

    if args.attention_for_uw_corruption:
        bert_attention = BertModel.from_pretrained("bert-base-uncased")
        bert_attention.resize_token_embeddings(len(tokenizer))
        model_attention = BertSelect(bert_attention)
        model_attention = load_model(model_attention, args.attention_model_path, 0, len(tokenizer))
        model_attention.to(device)
    else:
        model_attention = None

    raw_dd_train, raw_dd_dev = get_dd_corpus("train"), get_dd_corpus("validation")
    raw_dd_train = get_dd_corpus("train")
    raw_dd_dev = get_dd_corpus("validation")

    train_dataset = SelectionDataset(
        raw_dd_train,
        tokenizer,
        "train",
        300,
        args.retrieval_candidate_num,
        UTTR_TOKEN,
        "./data/selection/text_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
        "./data/selection/tensor_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
    )

    dev_dataset = SelectionDataset(
        raw_dd_dev,
        tokenizer,
        "dev",
        300,
        args.retrieval_candidate_num,
        UTTR_TOKEN,
        "./data/selection/text_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
        "./data/selection/tensor_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
    )
    print("Load end!")

    trainloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )
    validloader = DataLoader(dev_dataset, batch_size=args.batch_size, drop_last=True)

    """
    Training
    """
    crossentropy = CrossEntropyLoss()
    margin_loss_metric = torch.nn.MarginRankingLoss(margin=args.margin)
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
    )
    bce_metric = nn.BCELoss()
    writer = SummaryWriter(args.board_path)

    save_model(model, "begin", args.model_path)
    global_step = 0
    for epoch in range(args.epoch):
        print("Epoch {}".format(epoch))
        model.train()
        for step, batch in enumerate(tqdm(trainloader)):
            if step == 5:
                break
            optimizer.zero_grad()
            answer_ids, answer_mask = torch.tensor(batch[0]), torch.tensor(
                batch[args.retrieval_candidate_num]
            )
            assert len(batch) == 2 * args.retrieval_candidate_num + 1
            ids_list, mask_list, label = (
                batch[: args.retrieval_candidate_num],
                batch[args.retrieval_candidate_num : 2 * args.retrieval_candidate_num],
                batch[2 * args.retrieval_candidate_num],
            )
            bs = label.shape[0]

            ids_list = torch.cat(ids_list, 1).reshape(bs * args.retrieval_candidate_num, 300)
            mask_list = torch.cat(mask_list, 1).reshape(bs * args.retrieval_candidate_num, 300)

            # Input for auxiliary tasks
            corrupted_ids, corrupted_masks = corrupt_context_wordlevel_for_auxilary(
                answer_ids,
                answer_mask,
                use_attn=args.attention_for_uw_corruption,
                corrupt_ratio=args.uw_corrupt_ratio,
                sep_id=tokenizer.sep_token_id,
                skip_token_ids=tokens_to_skip,
                device=device,
                model=model_attention,
            )

            # Input for Usual training
            ids_list, mask_list = ids_list.to(device), mask_list.to(device)
            label = label.to(device)

            output, original_auxilary_output, corrupted_auxilary_output = model(
                ids_list,
                mask_list,
                answer_ids.to(device),
                answer_mask.to(device),
                corrupted_ids.to(device),
                corrupted_masks.to(device),
            )

            # Loss Calcuation
            if args.corrupt_loss_type == "margin":
                corrupt_loss = margin_loss_metric(
                    torch.ones(bs).to(device), original_auxilary_output, corrupted_auxilary_output
                )
            elif args.corrupt_loss_type == "crossentropy":
                corrupt_label = torch.tensor(
                    [1.0 for _ in range(len(original_auxilary_output))]
                    + [0.0 for _ in range(len(original_auxilary_output))]
                ).to(device)
                corrupt_loss = bce_metric(
                    torch.squeeze(torch.cat([original_auxilary_output, corrupted_auxilary_output])),
                    corrupt_label,
                )

            output = output.reshape(bs, -1)

            # Train step
            usual_loss = crossentropy(output, label)
            loss = usual_loss + args.alpha * corrupt_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            write2tensorboard(
                writer,
                {"select": usual_loss, "loss": loss, "corrupt": corrupt_loss},
                "train",
                global_step,
            )
            global_step += 1

        model.eval()
        lossdict = {"select": [], "loss": [], "corrupt": []}
        try:
            with torch.no_grad():
                for step, batch in enumerate(tqdm(validloader)):
                    answer_ids, answer_mask = torch.tensor(batch[0]), torch.tensor(
                        batch[args.retrieval_candidate_num]
                    )
                    assert len(batch) == 2 * args.retrieval_candidate_num + 1
                    ids_list, mask_list, label = (
                        batch[: args.retrieval_candidate_num],
                        batch[args.retrieval_candidate_num : 2 * args.retrieval_candidate_num],
                        batch[2 * args.retrieval_candidate_num],
                    )
                    bs = label.shape[0]

                    ids_list = torch.cat(ids_list, 1).reshape(
                        bs * args.retrieval_candidate_num, 300
                    )
                    mask_list = torch.cat(mask_list, 1).reshape(
                        bs * args.retrieval_candidate_num, 300
                    )

                    # Input for auxiliary tasks
                    corrupted_ids, corrupted_masks = corrupt_context_wordlevel_for_auxilary(
                        answer_ids,
                        answer_mask,
                        use_attn=args.attention_for_uw_corruption,
                        corrupt_ratio=args.uw_corrupt_ratio,
                        sep_id=tokenizer.sep_token_id,
                        skip_token_ids=tokens_to_skip,
                        device=device,
                        model=model_attention,
                    )

                    # Input for Usual training
                    ids_list, mask_list = ids_list.to(device), mask_list.to(device)
                    label = label.to(device)

                    output, original_auxilary_output, corrupted_auxilary_output = model(
                        ids_list,
                        mask_list,
                        answer_ids.to(device),
                        answer_mask.to(device),
                        corrupted_ids.to(device),
                        corrupted_masks.to(device),
                    )

                    # Loss Calcuation
                    if args.corrupt_loss_type == "margin":
                        corrupt_loss = margin_loss_metric(
                            torch.ones(bs).to(device),
                            original_auxilary_output,
                            corrupted_auxilary_output,
                        )
                    elif args.corrupt_loss_type == "crossentropy":
                        corrupt_label = torch.tensor(
                            [1.0 for _ in range(len(original_auxilary_output))]
                            + [0.0 for _ in range(len(original_auxilary_output))]
                        ).to(device)
                        corrupt_loss = bce_metric(
                            torch.squeeze(
                                torch.cat([original_auxilary_output, corrupted_auxilary_output])
                            ),
                            corrupt_label,
                        )

                    output = output.reshape(bs, -1)

                    # Train step
                    usual_loss = crossentropy(output, label)
                    loss = usual_loss + args.alpha * corrupt_loss
                    lossdict["select"].append(usual_loss.cpu().detach().numpy())
                    lossdict["loss"].append(loss.cpu().detach().numpy())
                    lossdict["corrupt"].append(corrupt_loss.cpu().detach().numpy())
            for k, v in lossdict.items():
                lossdict[k] = sum(v) / len(v)
            write2tensorboard(writer, lossdict, "valid", global_step)
        except Exception as err:
            print(err)

        save_model(model, epoch, args.model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="select_batch12_candi10",
    )
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument(
        "--attention_model_path", type=str, default="./logs/select_batch12_candi10_seed42/model"
    )
    parser.add_argument(
        "--retrieval_candidate_num", type=int, default=10, help="1개의 정답을 포함하여 몇 개의 candidate를 줄 것인지"
    )
    parser.add_argument("--random_initialization", type=str2bool, default=False)

    parser.add_argument(
        "--corrupt_loss_type", type=str, default="margin", choices=["margin", "crossentropy"]
    )
    parser.add_argument("--uw_corruption", type=str2bool, default=False)
    parser.add_argument("--ic_corruption", type=str2bool, default=False)
    parser.add_argument("--uw_corrupt_ratio", type=float, default=0.0)
    parser.add_argument("--attention_for_uw_corruption", type=str2bool, default=False)
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Auxiliary loss를 기존 loss에 얼마나 반영할지"
    )
    parser.add_argument(
        "--margin", type=float, default=0.5, help="marginal ranking loss를 쓸 때 margin value를 몇으로 할건지"
    )

    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    assert isinstance(args.uw_corruption, bool)
    assert isinstance(args.ic_corruption, bool)
    assert isinstance(args.attention_for_uw_corruption, bool)
    assert isinstance(args.random_initialization, bool)
    assert args.uw_corruption + args.ic_corruption >= 1

    args.exp_name = "select_batch{}_candi{}".format(args.batch_size, args.retrieval_candidate_num)
    if args.uw_corruption:
        args.exp_name += "-uw"
        if args.attention_for_uw_corruption:
            args.exp_name += "-attntop"
        else:
            args.exp_name += "-rand"
        args.exp_name += str(args.uw_corrupt_ratio)

    if args.corrupt_loss_type == "margin":
        args.exp_name += "-margin{}".format(args.margin)
    elif args.corrupt_loss_type == "crossentropy":
        args.exp_name += "-bce"
    else:
        raise ValueError

    args.exp_name += "-alpha{}".format(args.alpha)
    print(f"Exp name: {args.exp_name}")

    args.exp_path = os.path.join(args.log_path, args.exp_name)
    args.model_path = os.path.join(args.exp_path, "model")
    args.board_path = os.path.join(args.exp_path, "board")
    os.makedirs(args.model_path, exist_ok=False)
    os.makedirs(args.board_path, exist_ok=False)
    main(args)
