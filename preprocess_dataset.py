import json
import os
import zipfile
from typing import List, Union

import wget

from utils import get_uttr_token

DAILYDIALOG_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"


def download_dailydialog(daily_output_dir: str):
    os.makedirs("data", exist_ok=True)
    """Download the raw DailyDialog dataset
    Args:
        daily_output_dir (str): Path to save
    """
    dd_zip_fname = os.path.join(daily_output_dir, "ijcnlp_dailydialog.zip")
    if os.path.exists(dd_zip_fname):
        print("DailyDailog already downloaded!")
        return
    wget.download(DAILYDIALOG_URL, out=daily_output_dir)
    # Manually unzip the train/dev/test files

    dd_zip = zipfile.ZipFile(dd_zip_fname)
    dd_zip.extractall(daily_output_dir)

    for zipname in ["train", "validation", "test"]:
        dd_zip_file = zipfile.ZipFile(
            os.path.join(
                daily_output_dir,
                "ijcnlp_dailydialog/{}.zip".format(zipname),
            )
        )
        dd_zip_file.extractall(os.path.join(daily_output_dir, "ijcnlp_dailydialog/"))


def _read_txt_files(fname: str):
    assert os.path.exists(fname)
    with open(fname, "r") as f:
        ls = [el.strip().split("|||") for el in f.readlines()]
    return ls


def get_dd_corpus(setname):
    assert setname in ["train", "validation", "test"]
    fname = "./data/ijcnlp_dailydialog/{}/dialogues_{}.txt".format(setname, setname)
    assert os.path.exists(fname)
    with open(fname, "r") as f:
        ls = [el.strip() for el in f.readlines()]
        for idx, line in enumerate(ls):
            line = [
                el.strip().lower() for el in line.split("__eou__") if el.strip() != ""
            ]
            ls[idx] = line
    return ls


def get_dd_multiref_testset(dirname="./data/"):
    """
    Download datset released in https://arxiv.org/pdf/1907.10568.pdf

    RETURNS:
        List of [context(str), List of Multiple and appripriate responses]
    """
    if not os.path.exists(os.path.join(dirname, "multireftest.json")):
        URL = "https://github.com/prakharguptaz/multirefeval/raw/master/multiref-dataset/multireftest.json"
        wget.download(URL, out=os.path.join(dirname, "multireftest.json"))
    data = []
    with open(os.path.join(dirname, "multireftest.json"), "r") as f:
        for line in f.readlines():
            line = json.loads(line)
            assert line["fold"] == "test"
            data.append(line["dialogue"])
    UTTR_TOKEN = get_uttr_token()
    pairs = []
    for line in data:
        context = []
        assert isinstance(line, list) and all([isinstance(el, dict) for el in line])
        for turn_idx, single_turn in enumerate(line):
            if "responses" not in single_turn:
                assert turn_idx == len(line) - 1
                break
            uttr, responses = single_turn["text"], single_turn["responses"]
            assert len(responses) >= 5
            assert all([isinstance(el, str) for el in responses + [uttr]])
            context.append(uttr)
            pairs.append([UTTR_TOKEN.join(context), responses])
    return pairs


def main():
    download_dailydialog("./data/")
    res = get_dd_corpus("train")
    res = get_dd_corpus("validation")
    res = get_dd_corpus("test")
    res = get_dd_multiref_testset("./data/")


if __name__ == "__main__":
    main()
