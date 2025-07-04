# util functions about data

from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np
from numpy import percentile
import pandas as pd
import os
from util.consts import Tasks


def sensorGroup_to_xy(sensor_group: tuple, task: Tasks) -> tuple:
    """Based on task returns which column names are needed for
        x (input)
        y (output)
        next (if we should user next values for y).
        if y ==[] , we should use labels for y which is usually attack column.

    Args:
        sensor_group (tuple): given by findSensorActuator method
        task (Tasks): _description_

    Returns:
        tuple: (x: List ,y: List , next: boolean)
    """
    sensors, actuators, consts = sensor_group
    attack = ["attack"]
    x_string, task_string, y_string = task.name.split("_")
    xlist = []
    ylist = []
    next = task_string == "next"
    if x_string == "sacl":
        xlist = sensors + actuators + consts + attack
    else:
        if "s" in x_string:
            xlist += sensors
        if "a" in x_string:
            xlist += actuators
        if "c" in x_string:
            xlist += consts
        if "l" in x_string:
            xlist += attack

    if y_string == "all":
        ylist = sensors + actuators + consts
    else:
        if "s" in y_string:
            ylist += sensors
        if "a" in y_string:
            ylist += actuators
        if "c" in y_string:
            ylist += consts
        if "l" in y_string:
            ylist += attack
    return (xlist, ylist, next)


def get_attack_interval(attack):
    heads = []
    tails = []
    for i in range(len(attack)):
        if attack[i] == 1:
            if attack[i - 1] == 0:
                heads.append(i)

            if i < len(attack) - 1 and attack[i + 1] == 0:
                tails.append(i)
            elif i == len(attack) - 1:
                tails.append(i)
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))
    # print(heads, tails)
    return res


# calculate F1 scores
def eval_scores(scores, true_scores, th_steps, return_thresold=False):
    padding_list = [0] * (len(true_scores) - len(scores))
    # print(padding_list)

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method="ordinal")
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        fmeas[i] = f1_score(true_scores, cur_pred)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores) + 1))
        thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds
    return fmeas


def eval_mseloss(predicted, ground_truth):

    ground_truth_list = np.array(ground_truth)
    predicted_list = np.array(predicted)

    # mask = (ground_truth_list == 0) | (predicted_list == 0)

    # ground_truth_list = ground_truth_list[~mask]
    # predicted_list = predicted_list[~mask]

    # neg_mask = predicted_list < 0
    # predicted_list[neg_mask] = 0

    # err = np.abs(predicted_list / ground_truth_list - 1)
    # acc = (1 - np.mean(err))

    # return loss
    loss = mean_squared_error(predicted_list, ground_truth_list)

    return loss


def get_err_median_and_iqr(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr


def get_err_median_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage * 100)) - percentile(
        np_arr, int((1 - percentage) * 100)
    )

    return err_median, err_delta


def get_err_mean_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = trim_mean(np_arr, percentage)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage * 100)) - percentile(
        np_arr, int((1 - percentage) * 100)
    )

    return err_median, err_delta


def get_err_mean_and_std(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std


def get_f1_score(scores, gt, contamination):

    padding_list = [0] * (len(gt) - len(scores))
    # print(padding_list)

    threshold = percentile(scores, 100 * (1 - contamination))

    if len(padding_list) > 0:
        scores = padding_list + scores

    pred_labels = (scores > threshold).astype("int").ravel()

    return f1_score(gt, pred_labels)


def getAttacks(df: pd.DataFrame, label) -> np.ndarray:
    # r1 = df[(df[label] == 1) & (df[label].shift() == 0)].index
    # r2 = df[(df[label] == 1) & (df[label].shift(-1) == 0)].index

    attacks = df[df[label] == 1].copy(deep=True)
    count = attacks.shape[0]
    print("numebr of attacks = ", count)
    first_attack_index = attacks.first_valid_index()
    last_attack_index = attacks.last_valid_index()

    print("First attack index is : ", first_attack_index)
    print("Last attack index is : ", last_attack_index)

    attacks["count"] = attacks.index.to_series()
    attacks["count"] = attacks["count"] - attacks["count"].shift(
        fill_value=first_attack_index
    )
    attacks_starts = attacks[attacks["count"] != 1].index

    attacks["count"] = attacks.index.to_series()
    attacks["count"] = attacks["count"] - attacks["count"].shift(
        periods=-1, fill_value=last_attack_index
    )
    attacks_ends = attacks[attacks["count"] != -1].index
    assert len(attacks_starts) == len(attacks_ends)
    print("Number of attacks: ", len(attacks_starts))

    diff: np.ndarray = np.array([(b - a) for a, b in zip(attacks_starts, attacks_ends)])
    print("Minimum attack len: ", diff.min())
    print("Maximum attack len: ", diff.max())
    print("Mean attack len: ", diff.mean())

    return [[a, b] for a, b in zip(attacks_starts, attacks_ends)]


def createDummyDataset(samples: int, coulmns=3):
    csv_folder = "./data/dummy"
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    data_train = {
        "datetime": np.arange(1, samples + 1),
        "attack": np.zeros_like(samples),
    }
    data_test = {
        "datetime": np.arange(1, samples + 1),
        "attack": np.random.randint(0, 2, samples),
    }
    if os.path.exists(f"{csv_folder}/list.txt"):
        os.remove(f"{csv_folder}/list.txt")
    list_file = open(f"{csv_folder}/list.txt", "w")
    cols = [f"col_{col}" for col in range(coulmns)]
    list_file.write("\n".join(cols))
    list_file.close()
    for col in cols:
        data_train[col] = np.random.random(samples)
        data_test[col] = np.random.random(samples)

    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)

    # Save the DataFrame to a CSV file
    df_train.to_csv(f"{csv_folder}/train.csv", index=False)
    df_test.to_csv(f"{csv_folder}/test.csv", index=False)


if __name__ == "__main__":
    sg = (["s1", "s2", "s3", "s4"], ["a1", "a2", "a3"], ["c1", "c2", "c3"])
    l = sensorGroup_to_xy(sg, Tasks.sc_next_s)
    print(l)
