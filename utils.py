import json
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm


def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RankerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        setname: str,
        max_seq_len: int = 128,
        uttr_token: str = "[UTTR]",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.uttr_token = uttr_token
        assert setname in ["train", "dev", "test"]
        self.triplet_fname = "./data/triplet/triplet_{}.pck".format(setname)
        self.triplet_dataset = self._get_triplet_dataset(raw_dataset)
        self.tensor_fname = "./data/triplet/tensor_{}.pck".format(setname)
        self.feature = self._tensorize_triplet_dataset()

    def __len__(self):
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def _tensorize_triplet_dataset(self):
        if os.path.exists(self.tensor_fname):
            with open(self.tensor_fname, "rb") as f:
                return pickle.load(f)

        ids, masks, labels = [], [], []
        print("Tensorize...")
        for idx, triple in enumerate(tqdm(self.triplet_dataset)):
            assert len(triple) == 3 and all([isinstance(el, str) for el in triple])
            context, pos_uttr, neg_uttr = triple

            positive_sample = self.tokenizer(
                context,
                text_pair=pos_uttr,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            negative_sample = self.tokenizer(
                context,
                text_pair=neg_uttr,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids.extend(positive_sample["input_ids"])
            masks.extend(positive_sample["attention_mask"])
            labels.append(1)
            ids.extend(negative_sample["input_ids"])
            masks.extend(negative_sample["attention_mask"])
            labels.append(0)
        assert len(ids) == len(masks) == len(labels)
        data = torch.stack(ids), torch.stack(masks), torch.tensor(labels)
        with open(self.tensor_fname, "wb") as f:
            pickle.dump(data, f)
        return data

    def _get_triplet_dataset(self, raw_dataset):
        """
        [context,pos_response,negative_response]의 리스트를 반환합니다.
        이미 만들어서 저장해둔 파일이 있으면 그걸 가져다주고, 없으면 새로 만듬.

        Args:
            raw_dataset (List[List[str]]): List of conversation. Each conversation is list of utterance(str).
        """
        if os.path.exists(self.triplet_fname):
            print(f"{self.triplet_fname} exist!")
            with open(self.triplet_fname, "rb") as f:
                return pickle.load(f)

        triplet_dataset = self._make_triplet_dataset(raw_dataset)
        os.makedirs(os.path.dirname(self.triplet_fname), exist_ok=True)
        with open(self.triplet_fname, "wb") as f:
            pickle.dump(triplet_dataset, f)
        return triplet_dataset

    def _make_triplet_dataset(self, raw_dataset):
        """
        List of List of utterance인 raw_dataset을 받아서 (context, positive_response, negative_response (random)) 들의 리스트를 만들어서 반환
        """
        assert isinstance(raw_dataset, list) and all(
            [isinstance(el, list) for el in raw_dataset]
        )
        print(f"{self.triplet_fname} not exist. Make new file...")
        dataset = []
        all_responses = []
        for idx, conv in enumerate(tqdm(raw_dataset)):
            slided_conversation = self._slide_conversation(conv)
            dataset.extend(slided_conversation)
            all_responses.extend([el[1] for el in slided_conversation])
        for idx, el in enumerate(dataset):
            while True:
                sampled_random_negative = random.sample(all_responses, 1)[0]
                if sampled_random_negative != el[1]:
                    break
            dataset[idx].append(sampled_random_negative)
        return dataset

    def _slide_conversation(self, conversation):
        """
        multi-turn utterance로 이루어진 single conversation을 여러 개의 "context-response" pair로 만들어 반환
        """
        assert isinstance(conversation, list) and all(
            [isinstance(el, str) for el in conversation]
        )
        pairs = []
        for idx in range(len(conversation) - 1):
            context, response = conversation[: idx + 1], conversation[idx + 1]
            pairs.append([self.uttr_token.join(context), response])
        return pairs


def get_uttr_token():
    return "[UTTR]"


def dump_config(args):
    with open(os.path.join(args.exp_path, "config.json"), "w") as f:
        json.dump(vars(args), f)


def write2tensorboard(writer, value, setname, step):
    for k, v in value.items():
        writer.add_scalars(k, {setname: v}, step)
    writer.flush()


def save_model(model, epoch, model_path):
    torch.save(
        model.state_dict(),
        os.path.join(model_path, f"epoch-{epoch}.pth"),
    )


def load_model(model, model_path, epoch, len_tokenizer):
    model.resize_token_embeddings(len_tokenizer)
    model.load_state_dict(torch.load(model_path + f"/epoch-{epoch}.pth"))
    return model
