import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import RocCurveDisplay
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np
def fprat95tpr(ood_ps, ind_ps):
    """
    :param ood_ps
    :param ind_ps
    Find p-value threshold that results in 95% TPR. Then find FPR.
    :return:
    """

    sorted_ps = sorted(ood_ps) #from lowest to highest
    threshold = sorted_ps[int(len(ood_ps)*0.95)]
    thresholded = ind_ps<threshold
    return thresholded.mean()

def calibrated_detection_rate(ood_ps, ind_ps, tnr_threshold=1):
    """

    :param ood_ps:
    :param ind_ps:

    Essentially, fnr&0fpr
    :return:
    """
    sorted_ps = sorted(ind_ps)
    threshold = sorted_ps[int((1-tnr_threshold)*len(sorted_ps))] #minimum ind p_val that results in tnr of tnr_threshold
    # print(sorted_ps)
    # print("t:", threshold)
    return (tnr_threshold+(ood_ps<threshold).mean())/2

def auroc(ood_ps, ind_ps):
    true = [0]*len(ood_ps)+[1]*len(ind_ps)
    probs = list(ood_ps)+list(ind_ps)
    # probs = [i/max(ood_ps) for i in probs]

    auc = roc_auc_score(true, probs)
    return auc

def aupr(ood_ps, ind_ps):
    true = [0] * len(ood_ps) + [1] * len(ind_ps)
    probs = list(ood_ps) + list(ind_ps)
    # probs = [i/max(ood_ps) for i in probs]
    auc = average_precision_score(true, probs)
    return auc

# def sample_sensitivity():
#     """
#     Quantifies the sensitivity of a given method to sample size.
#     Essentially: area under the detection-error v sample size curve.
#     :return:
#     """

def correlation(ood_ps, ind_ps, ood_loss, ind_loss):
    merged_p = list(ood_ps)+list(ind_ps)
    merged_loss = list(ood_loss)+list(ind_loss)
    # sns.regplot(data=merged, x="P", y="loss")
    return spearmanr(merged_p, merged_loss)