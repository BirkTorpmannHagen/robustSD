import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import RocCurveDisplay
import seaborn as sns
from scipy.stats import spearmanr
def fprat95tpr(ood_dataframe, ind_dataframe):
    """
    :param ood_dataframe
    :param ind_dataframe
    Find p-value threshold that results in 95% TPR. Then find FPR.
    :return:
    """

    ood_ps = ood_dataframe["P"]
    sorted_ps = sorted(ood_ps) #from lowest to highest
    threshold = sorted_ps[int(len(ood_ps)*0.95)]
    thresholded = ind_dataframe["P"]<threshold
    return thresholded.mean()

def detection_error(ood_dataframe, ind_dataframe):
    pass

def auroc(ood_dataframe, ind_dataframe):
    true = [0]*len(ood_dataframe)+[1]*len(ind_dataframe)
    probs = list(ood_dataframe["P"])+list(ind_dataframe["P"])
    # probs = [i/max(ood_dataframe["P"]) for i in probs]
    auc = roc_auc_score(true, probs)
    return auc

def aupr(ood_dataframe, ind_dataframe):
    true = [0] * len(ood_dataframe) + [1] * len(ind_dataframe)
    probs = list(ood_dataframe["P"]) + list(ind_dataframe["P"])
    # probs = [i/max(ood_dataframe["P"]) for i in probs]
    auc = average_precision_score(true, probs)
    return auc

# def sample_sensitivity():
#     """
#     Quantifies the sensitivity of a given method to sample size.
#     Essentially: area under the detection-error v sample size curve.
#     :return:
#     """

def correlation(ood_dataframe, ind_dataframe):
    merged = pd.concat((ood_dataframe, ind_dataframe), axis=0)
    # sns.regplot(data=merged, x="P", y="loss")
    return spearmanr(merged["P"], merged["loss"])