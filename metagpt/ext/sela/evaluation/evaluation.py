from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score, root_mean_squared_error
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import matthews_corrcoef

def calculate_mcc(y_true, y_pred):
    """
    计算并返回 Matthews Correlation Coefficient (MCC)。

    参数:
    y_true (list 或 numpy.ndarray): 真实的二进制标签。
    y_pred (list 或 numpy.ndarray): 预测的二进制标签。

    返回:
    float: 计算得到的 MCC 值。
    """

    # 将输入转换为 NumPy 数组（如果它们是列表）
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    # 检查输入数据的长度是否一致
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 的长度必须一致。")

    # 计算 MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    return mcc

def calculate_auprc(y_true, y_scores):
    """
    计算并返回 Area Under the Precision-Recall Curve (AUPRC)。
    
    参数:
    y_true (list 或 numpy.ndarray): 真实的二进制标签。
    y_scores (list 或 numpy.ndarray): 预测的分数或概率。
    
    返回:
    float: 计算得到的 AUPRC 值。
    """
    
    # 将输入转换为 NumPy 数组（如果它们是列表）
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    
    if isinstance(y_scores, list):
        y_scores = np.array(y_scores)
    
    # 检查输入数据的长度是否一致
    if len(y_true) != len(y_scores):
        raise ValueError("y_true 和 y_scores 的长度必须一致。")
    
    # 计算 Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    # 计算 AUPRC
    auprc = auc(recall, precision)
    
    return auprc



def monthly_ks_difference(pred, ground_truth):
    # 合并两个DataFrame，假设它们有相同的行数和顺序
    data = pd.DataFrame({
        'dt': pred['dt'],
        'pred_label': pred['target'],
        'true_label': ground_truth['label']
    })
    
    # 获取所有唯一的月份
    unique_months = data['dt'].unique()
    
    # 用于存储每个月的KS值
    ks_values = []
    
    for month in unique_months:
        # 筛选出当前月份的数据
        monthly_data = data[data['dt'] == month]
        
        # 计算KS统计量
        ks_statistic, _ = ks_2samp(monthly_data['pred_label'], monthly_data['true_label'])
        
        # 添加到列表中
        ks_values.append(ks_statistic)
    
    # 返回最大KS值和最小KS值的差
    return max(ks_values) - min(ks_values)

# 示例用法
# pred = pd.DataFrame({'dt': [...], 'label': [...]})
# ground_truth = pd.DataFrame({'dt': [...], 'label': [...]})
# result = monthly_ks_difference(pred, ground_truth)
# print("Difference between max and min KS: ", result)


def ks_compute(target_arr, proba_arr):
    '''
    ----------------------------------------------------------------------
    功能：利用scipy库函数计算ks指标
    ----------------------------------------------------------------------
    :param proba_arr:  numpy array of shape (1,), 预测为1的概率.
    :param target_arr: numpy array of shape (1,), 取值为0或1.
    ----------------------------------------------------------------------
    :return ks_value: float, ks score estimation
    ----------------------------------------------------------------------
    示例：
    >>> ks_compute(target_arr=df[target], proba_arr=df['score'])
    >>> 0.5262199213881699
    ----------------------------------------------------------------------
    '''
    proba_arr = np.array(proba_arr)
    target_arr = np.array(target_arr)
    from scipy.stats import ks_2samp
    get_ks = lambda proba_arr, target_arr: ks_2samp(proba_arr[target_arr == 1], \
                                           proba_arr[target_arr == 0]).statistic
    ks_value = get_ks(proba_arr, target_arr)
    return ks_value


def evaluate_score(pred, gt, metric):
    if metric == "accuracy":
        return accuracy_score(gt, pred)
    elif metric == "f1":
        unique_classes = sorted(list(np.unique(gt)))
        if 1 in unique_classes and 0 in unique_classes:
            pos_label = 1
        else:
            pos_label = unique_classes[0] if len(unique_classes) == 2 else None
        return f1_score(gt, pred, pos_label=pos_label)
    elif metric == "f1 weighted":
        return f1_score(gt, pred, average="weighted")
    elif metric == "roc_auc":
        return roc_auc_score(gt, pred)
    elif metric == "rmse":
        return root_mean_squared_error(gt, pred)
        # return mean_squared_error(gt, pred, squared=False)
    elif metric == "log rmse":
        # return mean_squared_error(np.log1p(gt), np.log1p(pred), squared=False)
        return root_mean_squared_error(np.log1p(gt), np.log1p(pred))
    elif metric == "ks_compute":
        return ks_compute(gt, pred)
    elif metric == "calculate_auprc":
        return calculate_auprc(gt, pred)
    elif metric == "calculate_mcc":
        return calculate_mcc(gt, pred)
    elif metric == "monthly_ks_difference":
        return 3.0 - monthly_ks_difference(pred, gt)
    else:
        raise ValueError(f"Metric {metric} not supported")


def node_evaluate_score_sela(node):
    preds = node.get_and_move_predictions("test")["target"]
    gt = node.get_gt("test")["target"]
    metric = node.state["dataset_config"]["metric"]
    return evaluate_score(preds, gt, metric)


def node_evaluate_score_mlebench(node):
    # TODO
    from mlebench.grade import grade_csv
    from mlebench.registry import registry

    competition_id = node.state["task"]
    data_dir = Path(node.state["custom_dataset_dir"]).parent.parent.parent  # prepared/public/../../../
    pred_path = node.get_predictions_path("test")
    new_registry = registry.set_data_dir(data_dir)
    competition = new_registry.get_competition(competition_id)
    submission = Path(pred_path)
    report = grade_csv(submission, competition).to_dict()
    report["submission_path"] = str(submission)
    return report
