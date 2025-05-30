from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import torch
from util.consts import Tasks
from parameters.StatisticalParameters import StatisticalParameters
from parameters.MetricsParameters import MetricsParameters
from torchmetrics.classification import BinaryConfusionMatrix


class BaseThreshold:
    """docstring for BaseThreshold."""

    def __init__(self):
        super(BaseThreshold, self).__init__()

    def fit(self, loss: torch.Tensor):
        pass

    def transform(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def loadFromPath(self, path):
        pass

    def save(self, path):
        pass

    def summary(self):
        return "This is the best class for defining threshold for classification."


class IqrThreshold(BaseThreshold):
    """docstring for IqrThreshold."""

    def __init__(self):
        super(IqrThreshold, self).__init__()
        self.stats: StatisticalParameters = None

    def fit(self, losses: torch.Tensor):
        self.stats = createIrqStats(losses)

    def transform(self, loss: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predicted_labels = (
                ((loss - self.stats.medians).abs() / self.stats.iqr).max(-1).values
            )
            preds = torch.where(
                predicted_labels > self.stats.threshold,
                1,
                0,
            )
        return preds

    def summary(self):
        return "IqrThreshold:\n\n" + self.stats.summary()


class MaxLossThreshold(BaseThreshold):
    """docstring for MaxLossThreshold."""

    def __init__(self):
        super(MaxLossThreshold, self).__init__()
        self.max = 0.0

    def fit(self, losses: torch.Tensor):
        _loss = losses.clone()
        if _loss.dim() == 2:
            _loss = _loss.sum(-1)
        self.max = _loss.max()

    def transform(self, loss: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predicted_labels = (
                ((loss - self.stats.medians).abs() / self.stats.iqr).max(-1).values
            )
            acu_loss += predicted_labels.sum()
            preds = torch.where(
                predicted_labels > self.stats.threshold,
                1,
                0,
            )
        return preds


class MyConfusuion(BinaryConfusionMatrix):
    """docstring for MyConfusuion."""

    def __init__(self, *args, thr: BaseThreshold, **kwargs):
        super(MyConfusuion, self).__init__(*args, **kwargs)
        self.thr = thr

    def update(self, loss: torch.Tensor, target: torch.Tensor) -> None:
        super().update(self.thr.transform(loss), target)


def get_full_err_scores(test_result, val_result):
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    all_scores = None
    all_normals = None
    feature_num = np_test_result.shape[-1]

    labels = np_test_result[2, :, 0].tolist()

    for i in range(feature_num):
        test_re_list = np_test_result[:2, :, i]
        val_re_list = np_val_result[:2, :, i]

        scores = get_err_scores(test_re_list, val_re_list)
        normal_dist = get_err_scores(val_re_list, val_re_list)

        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((all_scores, scores))
            all_normals = np.vstack((all_normals, normal_dist))

    return all_scores, all_normals


def get_final_err_scores(test_result, val_result):
    full_scores, all_normals = get_full_err_scores(
        test_result, val_result, return_normal_scores=True
    )

    all_scores = np.max(full_scores, axis=0)

    return all_scores


def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(
        np.subtract(
            np.array(test_predict).astype(np.float64),
            np.array(test_gt).astype(np.float64),
        )
    )
    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num : i + 1])

    return smoothed_err_scores


def get_loss(predict, gt):
    return eval_mseloss(predict, gt)


def get_f1_scores(total_err_scores, gt_labels, topk=1):
    print("total_err_scores", total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    topk_err_score_map = []
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):

        sum_score = sum(
            score
            for k, score in enumerate(
                sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])
            )
        )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas


def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    total_topk_err_scores = []
    topk_err_score_map = []

    total_topk_err_scores = np.sum(
        np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0
    )

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, thresold


def get_best_performance_data(total_err_scores, gt_labels, topk=1):

    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    total_topk_err_scores = []
    topk_err_score_map = []

    total_topk_err_scores = np.sum(
        np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0
    )

    final_topk_fmeas, thresolds = eval_scores(
        total_topk_err_scores, gt_labels, 400, return_thresold=True
    )

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), pre, rec, auc_score, thresold


def createIrqStats(all_losses) -> StatisticalParameters:

    medians = all_losses.median(0).values
    q = torch.quantile(
        all_losses, torch.tensor([0.25, 0.75], device=all_losses.device), dim=0
    )
    q = q[1] - q[0]
    # TODO: 25/04/05 19:23:11 Instead of returning just one max, we can return maximum for every node
    threshold = ((all_losses - medians).abs() / q).max()

    return StatisticalParameters(medians, q, threshold)
