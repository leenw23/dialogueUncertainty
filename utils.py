import json
import os
import pickle
import random
import re
import numpy as np
import torch
from tqdm import tqdm


def make_corrupted_select_dataset(
    uw_data,
    dd_dataset,
    retrieval_candidate_num,
    save_fname,
    tokenizer,
    max_seq_len,
    replace_golden_to_nota,
):
    assert not replace_golden_to_nota
    if os.path.exists(save_fname):
        print("{} exist!".format(save_fname))
        with open(save_fname, "rb") as f:
            return pickle.load(f)
    nota_token = get_nota_token()
    assert isinstance(uw_data, list) and all([len(el) == 2 for el in uw_data])
    responses = [uttr for conv in dd_dataset for uttr in conv[1:]]
    assert all([isinstance(el, str) for el in responses])
    for idx, hist in enumerate(uw_data):
        assert len(hist) == 2 and all([isinstance(el, str) for el in hist])
        assert hist[1] == nota_token or not replace_golden_to_nota
        candidates = random.sample(responses, retrieval_candidate_num - 1)
        uw_data[idx].extend(candidates)

    ids_list = [[] for _ in range(retrieval_candidate_num)]
    masks_list = [[] for _ in range(retrieval_candidate_num)]
    labels = []
    print("Tensorize...")
    for sample_idx, sample in enumerate(tqdm(uw_data)):
        assert len(sample) == 1 + retrieval_candidate_num
        assert all([isinstance(el, str) for el in sample])
        context, candidates = sample[0], sample[1:]
        assert len(candidates) == retrieval_candidate_num
        encoded = tokenizer(
            [context] * retrieval_candidate_num,
            text_pair=candidates,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded_ids, encoded_mask = encoded["input_ids"], encoded["attention_mask"]
        assert len(encoded_ids) == len(encoded_mask) == retrieval_candidate_num
        for candi_idx in range(retrieval_candidate_num):
            ids_list[candi_idx].append(encoded_ids[candi_idx])
            masks_list[candi_idx].append(encoded_mask[candi_idx])
        labels.append(0)
    assert len(list(set([len(el) for el in ids_list]))) == 1
    assert len(list(set([len(el) for el in masks_list]))) == 1
    ids_list = [torch.stack(el) for el in ids_list]
    masks_list = [torch.stack(el) for el in masks_list]
    labels = torch.tensor(labels)
    data = ids_list + masks_list + [labels]
    assert len(data) == 1 + 2 * retrieval_candidate_num
    with open(save_fname, "wb") as f:
        pickle.dump(data, f)
    return data


def make_tuple(exp):
    assert "(" in exp and ")" in exp and exp.count(",") == 1
    exp = [el.strip() for el in exp.strip()[1:-1].split(",")]
    return exp


def get_ic_annotation(fname, change_ic_to_original: bool, is_dev):
    with open(fname, "r") as f:
        ls = [el.strip() for el in f.readlines()]

    item_list, item = [], {}
    uttr_token = get_uttr_token()

    for line in ls:
        if line == "":
            assert len(item) != 0
            item_list.append(item)
            item = {}
            continue
        if len(item) == 0:
            tmp = [int(el) for el in line.strip().split()]
            item["removed_context_num"] = tmp[1]
            item["remain_context_num"] = tmp[2]
            continue
        if "uttrs" not in item:
            item["uttrs"] = []
        item["uttrs"].append(line)

    final_output = []
    for item in item_list:
        removed_num, remain_num = item["removed_context_num"], item["remain_context_num"]
        uttrs = item["uttrs"]
        assert len(uttrs) == removed_num + remain_num + 1
        context = uttrs[:-1]
        response = uttrs[-1]
        if not change_ic_to_original:
            context = context[removed_num:]
            assert len(context) == remain_num
        else:
            assert len(context) == remain_num + removed_num
        context = uttr_token.join(context)
        context = context.replace(" ##", "")
        response = response.replace(" ##", "")
        assert "##" not in context
        assert "##" not in response
        final_output.append([context, response])
    return final_output


def get_uw_annotation(fname, change_uw_to_original: bool, replace_golden_to_nota: bool, is_dev):
    with open(fname, "r") as f:
        ls = [el.strip() for el in f.readlines()]
    item_list, item = [], {}
    uttr_token = get_uttr_token()

    for line in ls:
        if line == "":
            assert len(item) != 0
            item_list.append(item)
            item = {}
            continue
        # head
        if len(item) == 0:
            idx, change_num = [int(el) for el in line.split()]
            item["idx"] = idx
            item["num_change"] = change_num
            continue
        # original
        if len(item) == 2:
            original_words = line.split()
            item["original_words"] = original_words
            continue
        # original
        if len(item) == 3:
            changed_words = line.split()
            item["changed_words"] = changed_words
            continue
        if len(item) == 4:
            assert "uttrs" not in item
            item["uttrs"] = []
        item["uttrs"].append(line.strip())

    final_output = []
    for itemIdx, item in enumerate(item_list):
        change_num, uttrs, org_words, chd_words = (
            item["num_change"],
            item["uttrs"],
            item["original_words"],
            item["changed_words"],
        )
        assert len(org_words) == len(chd_words)
        context, response = uttr_token.join(uttrs[:-1]), uttrs[-1]
        if change_uw_to_original:
            total_restore_count = 0
            for word_idx, word in enumerate(chd_words):
                total_restore_count += context.count(word)
                context = context.replace(word, org_words[word_idx])
            try:
                assert total_restore_count == change_num
            except:
                continue
        context = context.replace(" ##", "")
        response = response.replace(" ##", "")
        assert "##" not in context
        assert "##" not in response
        final_output.append([context, response])

    if is_dev:
        return final_output[: int(len(final_output) * 0.3)]
    else:
        return final_output[int(len(final_output) * 0.3) :]


def get_uw_annotation_legacy(
    fname, change_uw_to_original: bool, replace_golden_to_nota: bool, is_dev
):
    with open(fname, "r") as f:
        ls = [el.strip() for el in f.readlines()]
    item_list, item = [], []
    uttr_token = get_uttr_token()
    nota_token = get_nota_token()
    for line in ls:
        if line == "":
            if len(item) != 0:
                item_list.append(item)
                item = []
            continue

        if "(" in line and ")" in line:
            parsed_tuple = re.findall(r"\([^()]*\)", line)
            num_change = int(line.strip().split()[-1])
            change_map = [make_tuple(el) for el in parsed_tuple]
            assert len(parsed_tuple) == len(change_map)
            item.append(change_map)
            continue
        item.append(line)

    final_output = []
    for item_idx, item in enumerate(item_list):
        change_map, uttrs = item[0], item[1:]
        context = uttr_token.join(uttrs[:-1])
        response = uttrs[-1] if not replace_golden_to_nota else nota_token
        error_case = False
        if change_uw_to_original:
            for change_history in change_map:
                org, chd = change_history
                try:
                    assert chd in context or chd[0].upper() + chd[1:] in context
                except:
                    error_case = True
                    break
                context = context.replace(chd, org).replace(chd[0].upper() + chd[1:], org)
        if not error_case:
            final_output.append([context, response])

    if is_dev:
        return final_output[: int(len(final_output) * 0.3)]
    else:
        return final_output[int(len(final_output) * 0.3) :]


# def read_uw_annotation(fname, replace_golden_to_nota):
#     nota_token = get_nota_token()
#     with open(fname, "r") as f:
#         ls = [el.strip() for el in f.readlines()]
#     sample = []
#     tmp = []
#     for line in ls:
#         if line == "":
#             if tmp == []:
#                 continue
#             context = "[UTTR]".join(tmp[1:-1])
#             if replace_golden_to_nota:
#                 response = nota_token
#             else:
#                 response = tmp[-1]
#             sample.append([context, response])
#             tmp = []
#             continue
#         tmp.append(line)
#     if tmp != []:
#         context = "[UTTR]".join(tmp[1:-1])
#         if replace_golden_to_nota:
#             response = nota_token
#         else:
#             response = tmp[-1]
#         sample.append([context, response])
#     return sample


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def recall_x_at_k(score_list, x, k, answer_index):
    """
    R_x@K를 계산합니다.
    """
    assert len(score_list) == x
    sorted_score_index = np.array(score_list).argsort()[::-1]
    assert answer_index in sorted_score_index
    return int(answer_index in sorted_score_index[:k])


class SelectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        setname: str,
        max_seq_len: int = 300,
        num_candidate: int = 10,
        uttr_token: str = "[UTTR]",
        txt_save_fname: str = None,
        tensor_save_fname: str = None,
        corrupted_context_dataset=None,
        # add_nota_in_every_candidate=False,
    ):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.uttr_token = uttr_token
        assert setname in ["train", "dev", "test"]
        txt_save_fname, tensor_save_fname = (
            txt_save_fname.format(setname),
            tensor_save_fname.format(setname),
        )
        # self.add_nota = add_nota_in_every_candidate
        selection_dataset = self._get_selection_dataset(
            raw_dataset, num_candidate, txt_save_fname, corrupted_context_dataset
        )
        # if self.add_nota:
        #    for el in selection_dataset:
        #        assert "[NOTA]" in el
        self.feature = self._tensorize_selection_dataset(
            selection_dataset, tensor_save_fname, num_candidate
        )

    def __len__(self):
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def _tensorize_selection_dataset(self, selection_dataset, tensor_save_fname, num_candidate):
        if os.path.exists(tensor_save_fname):
            with open(tensor_save_fname, "rb") as f:
                return pickle.load(f)
        ids_list = [[] for _ in range(num_candidate)]
        masks_list = [[] for _ in range(num_candidate)]
        labels = []
        print("Tensorize...")
        for sample_idx, sample in enumerate(tqdm(selection_dataset)):
            assert len(sample) == 1 + num_candidate and all([isinstance(el, str) for el in sample])
            context, candidates = sample[0], sample[1:]
            assert len(candidates) == num_candidate

            encoded = self.tokenizer(
                [context] * num_candidate,
                text_pair=candidates,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoded_ids, encoded_mask = encoded["input_ids"], encoded["attention_mask"]
            assert len(encoded_ids) == len(encoded_mask) == num_candidate
            for candi_idx in range(num_candidate):
                ids_list[candi_idx].append(encoded_ids[candi_idx])
                masks_list[candi_idx].append(encoded_mask[candi_idx])
            labels.append(0)

        assert len(list(set([len(el) for el in ids_list]))) == 1
        assert len(list(set([len(el) for el in masks_list]))) == 1
        ids_list = [torch.stack(el) for el in ids_list]
        masks_list = [torch.stack(el) for el in masks_list]
        labels = torch.tensor(labels)
        data = ids_list + masks_list + [labels]
        assert len(data) == 1 + 2 * num_candidate
        with open(tensor_save_fname, "wb") as f:
            pickle.dump(data, f)
        return data

    def _get_selection_dataset(
        self, raw_dataset, num_candidate, txt_save_fname, corrupted_context_dataset
    ):
        """Selection evlauation을 위한 데이터셋이 이미 만들어져있으면 가져오고 아니면 새로 만들어서 저장하고 가져옴.

        Args:
            raw_dataset ([type]): Raw DailyDialog Testset
            num_candidate ([type]): 총 몇 개의 candidate중에 정답을 찾도록 할 것인지. Recall_x@k에서의 X
            txt_save_fname ([type]): 만든 결과의 text를 저장할 파일
            corrupted_context_dataset: Context 일부가 고장난 데이터셋. Propose Method (maybe?)

        Returns:
            [type]: [description]
        """

        print("Selection filename: {}".format(txt_save_fname))
        if os.path.exists(txt_save_fname):
            print(f"{txt_save_fname} exist!")
            with open(txt_save_fname, "rb") as f:
                return pickle.load(f)

        selection_dataset = self._make_selection_dataset(
            raw_dataset, num_candidate, corrupted_context_dataset
        )
        os.makedirs(os.path.dirname(txt_save_fname), exist_ok=True)
        with open(txt_save_fname, "wb") as f:
            pickle.dump(selection_dataset, f)
        return selection_dataset

    def _make_selection_dataset(self, raw_dataset, num_candidate, corrupted_context_dataset):
        """num_candidate개의 response candidate를 가진 샘플들을 만들어서 반환.
        Returns:
            datset: List of [context(str), positive_response(str), negative_response_1(str), (...) negative_response_(num_candidate-1)(str)]
        """
        assert isinstance(raw_dataset, list) and all([isinstance(el, list) for el in raw_dataset])
        print(f"Serialized selection not exist. Make new file...")
        dataset = []
        all_responses = []
        for idx, conv in enumerate(tqdm(raw_dataset)):
            slided_conversation = self._slide_conversation(conv)
            # Check the max sequence length
            for single_conv in slided_conversation:
                assert len(single_conv) == 2 and all([isinstance(el, str) for el in single_conv])
                concat_single_conv = " ".join(single_conv)
                if len(self.tokenizer.tokenize(concat_single_conv)) + 3 <= 300:
                    dataset.append(single_conv)
            all_responses.extend([el[1] for el in slided_conversation])

        if corrupted_context_dataset is not None:
            print("Samples with corrupted context are also included in training")
            print("Before: {}".format(len(dataset)))
            half_sampled_corrupt_sample = random.sample(
                corrupted_context_dataset, int(len(dataset) / 2)
            )
            for corrupted_sample in tqdm(half_sampled_corrupt_sample):
                changed_context = self.tokenizer.decode(corrupted_sample["changed_context"]).strip()
                assert isinstance(changed_context, str)
                assert "[CLS]" == changed_context[:5]
                assert "[SEP]" == changed_context[-5:]
                tmp_text = changed_context[5:-5].strip()
                assert len(self.tokenizer.tokenize(tmp_text)) + 2 <= 300
                dataset.append([tmp_text, "[NOTA]"])
            print("After: {}".format(len(dataset)))

        for idx, el in enumerate(dataset):
            sampled_random_negative = random.sample(all_responses, num_candidate)
            if el[1] in sampled_random_negative:
                sampled_random_negative.remove(el[1])
            sampled_random_negative = sampled_random_negative[: num_candidate - 1]
            dataset[idx].extend(sampled_random_negative)

            # if not self.add_nota:
            #     sampled_random_negative = sampled_random_negative[: num_candidate - 1]
            #     dataset[idx].extend(sampled_random_negative)
            # else:
            #     sampled_random_negative = ["[NOTA]"] + sampled_random_negative[: num_candidate - 2]
            #     dataset[idx].extend(sampled_random_negative)
            assert len(dataset[idx]) == 1 + num_candidate
            assert all([isinstance(txt, str) for txt in dataset[idx]])
        return dataset

    def _slide_conversation(self, conversation):
        """
        multi-turn utterance로 이루어진 single conversation을 여러 개의 "context-response" pair로 만들어 반환
        """
        assert isinstance(conversation, list) and all([isinstance(el, str) for el in conversation])
        pairs = []
        for idx in range(len(conversation) - 1):
            context, response = conversation[: idx + 1], conversation[idx + 1]
            pairs.append([self.uttr_token.join(context), response])
        return pairs


class RankerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        setname: str,
        max_seq_len: int = 300,
        uttr_token: str = "[UTTR]",
        tensor_fname: str = None,
        corrupted_dataset=None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.uttr_token = uttr_token
        self.corrupted_dataset = corrupted_dataset
        assert setname in ["train", "dev", "test"]
        self.triplet_fname = "./data/triplet/triplet_{}.pck".format(setname)
        self.triplet_dataset = self._get_triplet_dataset(raw_dataset)
        if tensor_fname is None:
            self.tensor_fname = "./data/triplet/tensor_{}.pck".format(setname)
        else:
            self.tensor_fname = tensor_fname.format(setname)
        self.feature = self._tensorize_triplet_dataset(corrupted_dataset)

    def __len__(self):
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def _tensorize_triplet_dataset(self, corrupted_dataset):
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
        print("Triplet filename: {}".format(self.triplet_fname))
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
        assert isinstance(raw_dataset, list) and all([isinstance(el, list) for el in raw_dataset])
        print(f"{self.triplet_fname} not exist. Make new file...")
        dataset = []
        all_responses = []
        for idx, conv in enumerate(tqdm(raw_dataset)):
            slided_conversation = self._slide_conversation(conv)
            # Check the max sequence length
            for single_conv in slided_conversation:
                assert len(single_conv) == 2 and all([isinstance(el, str) for el in single_conv])
                concat_single_conv = " ".join(single_conv)
                if len(self.tokenizer.tokenize(concat_single_conv)) + 3 <= 300:
                    dataset.append(single_conv)
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
        assert isinstance(conversation, list) and all([isinstance(el, str) for el in conversation])
        pairs = []
        for idx in range(len(conversation) - 1):
            context, response = conversation[: idx + 1], conversation[idx + 1]
            pairs.append([self.uttr_token.join(context), response])
        return pairs


def get_uttr_token():
    return "[UTTR]"


def get_nota_token():
    return "[NOTA]"


def dump_config(args):
    with open(os.path.join(args.exp_path, "config.json"), "w") as f:
        json.dump(vars(args), f)


def write2tensorboard(writer, value, setname, step):
    for k, v in value.items():
        writer.add_scalars(k, {setname: v}, step)
    writer.flush()


def save_model(model, epoch, model_path):
    try:
        torch.save(
            model.module.state_dict(),
            os.path.join(model_path, f"epoch-{epoch}.pth"),
        )
    except:
        torch.save(
            model.state_dict(),
            os.path.join(model_path, f"epoch-{epoch}.pth"),
        )


def load_model(model, model_path, epoch, len_tokenizer):
    model.bert.resize_token_embeddings(len_tokenizer)
    model.load_state_dict(torch.load(model_path + f"/epoch-{epoch}.pth"))
    return model


def make_random_negative_for_multi_ref(multiref_original, num_neg=30):
    """multi-candidate(positive)가 있는 데이터셋에 대해, 같은 수의 random-negative response를 찾아서 반환

    Args:
        multiref_original (List[context(str), List[postiive_utterance(str)]]): get_dd_multiref_testset()에서 반환된 multi-reference candidates
        num_neg (int, optional): 몇 개의 random response를 담아서 반환할건지.
    Returns:
        multiref_original (List[context(str), List[postiive_utterance(str),List[negative_utterance(str)]]):

    """
    for idx, item in enumerate(multiref_original):
        context, responses = item
        sample = random.sample(range(len(multiref_original)), num_neg + 1)
        if idx in sample:
            sample.remove(idx)
        else:
            sample = sample[:-1]
        responses = [multiref_original[sample_idx][1] for sample_idx in sample]
        responses = [el for el1 in responses for el in el1]
        assert all([isinstance(el, str) for el in responses])
        negative = random.sample(responses, num_neg)
        multiref_original[idx].append(negative)
    return multiref_original
