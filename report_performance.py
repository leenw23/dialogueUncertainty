import json, pickle
from utils import recall_x_at_k
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score_metric
import os
from pprint import pprint
import calibration as cal


def main_script(dirname):
    fnames = sorted([el for el in os.listdir(dirname) if ".json" in el])
    for fname in fnames:
        main(os.path.join(dirname, fname))


def softmax_np(logits):
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    return [float(el) for el in probs]


def main(fname):
    assert ".json" in fname
    output_fname = fname.replace(".json", ".txt")
    if os.path.exists(output_fname):
        return
    with open(fname, "r") as f:
        prediction_data = [json.loads(el) for el in f.readlines() if el.strip() != ""]

    r10 = run_origianl_recall(prediction_data, 10)d
    calibration_error = cal.get_ece(
        [softmax_np(l["pred"]) for l in prediction_data], [0 for _ in range(len(prediction_data))]
    )
    print(fname)
    print("R10@1: {}".format(r10))
    print("ECE-R10@1: {}".format(calibration_error))

    # assert not os.path.exists(output_fname)
    with open(output_fname, "w") as f:
        f.write("R10@1: {}\n".format(r10))
        f.write("ECE-R10@1: {}\n".format(calibration_error))


def run_origianl_recall(
    prediction_list,
    x: int,
):
    """Uncertainty threshold를 기반으로 Evaluation을 진행합니다.

    Args:
        prediction_list (List[Dict[str,Union[List[float], bool]]]): {
                    "pred": [list of unnormalized score],
                    "uncertainty": [list of unnormalized uncertainty],
                    "is_uw": bool,
                }
    """
    Recall_list = []
    for item in prediction_list:
        uncertainty = item["uncertainty"][:x]
        prediction_outcome = item["pred"][:x]
        Recall_list.append(recall_x_at_k(prediction_outcome, x, 1, 0))
    return sum(Recall_list) / len(Recall_list)


if __name__ == "__main__":
    dirname = "./ic_filtered_experiment_results/"
    main_script(dirname)
