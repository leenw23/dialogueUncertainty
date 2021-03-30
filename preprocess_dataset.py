import json
import os
import wget
import zipfile
from typing import List, Union


DAILYDIALOG_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"


def download_dailydialog(daily_output_dir: str):
    os.makedirs("data", exist_ok=True)
    """Download the raw DailyDialog dataset
    Args:
        daily_output_dir (str): Path to save
    """
    wget.download(DAILYDIALOG_URL, out=daily_output_dir)
    # Manually unzip the train/dev/test files

    dd_zip = zipfile.ZipFile(
        os.path.join(daily_output_dir, "ijcnlp_dailydialog.zip")
    )
    dd_zip.extractall(daily_output_dir)

    for zipname in ["train", "validation", "test"]:
        dd_zip_file = zipfile.ZipFile(
            os.path.join(
                daily_output_dir, "ijcnlp_dailydialog/{}.zip".format(zipname)
            )
        )
        dd_zip_file.extractall(
            os.path.join(daily_output_dir, "ijcnlp_dailydialog/")
        )


def _read_txt_files(fname: str):
    assert os.path.exists(fname)
    with open(fname, "r") as f:
        ls = [el.strip().split("|||") for el in f.readlines()]
    return ls


def get_dd_corpus(setname):
    assert setname in ["train", "validation", "test"]
    fname = "./data/ijcnlp_dailydialog/{}/dialogues_{}.txt".format(
        setname, setname
    )
    assert os.path.exists(fname)
    with open(fname, "r") as f:
        ls = [el.strip() for el in f.readlines()]
        for idx, line in enumerate(ls):
            line = [
                el.strip().lower()
                for el in line.split("__eou__")
                if el.strip() != ""
            ]
            ls[idx] = line
    return ls


def main():
    download_dailydialog("./data/")
    res = get_dd_corpus("train")
    res = get_dd_corpus("validation")
    res = get_dd_corpus("test")


if __name__ == "__main__":
    main()