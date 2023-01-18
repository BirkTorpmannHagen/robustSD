import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import RocCurveDisplay
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
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

def correlation_split(ood_ps, ind_ps, ood_loss, ind_loss):
    merged_p = list(ood_ps)+list(ind_ps)
    merged_loss = list(ood_loss)+list(ind_loss)
    # sns.regplot(data=merged, x="P", y="loss")
    return spearmanr(merged_p, merged_loss)

def correlation(ps, loss):

    return spearmanr(ps, loss)

def get_loss_pdf_from_ps(ps, loss, test_ps, test_losses, bins=100):
    """
        Computes a pdf for the given number of bins, and gets the likelihood of the test loss at the given test_ps bin.
        #todo: collect a new noise dataset with the right predictor
        :returns the average likelihood of the observed test-loss as bootstrapped from the pdf w/noise.
        Higher likelihood ~ more likely that the model is correct more often.
        
    """

    pargsort = np.argsort(ps)
    sorted_ps = np.array(ps)[pargsort]
    sorted_loss = np.array(loss)[pargsort]
    p_bins = sorted_ps[::len(sorted_ps)//bins]
    loss_samples_per_bin = [sorted_loss[i:j] for i, j in zip(range(0, len(sorted_loss), len(sorted_loss)//bins),
                                                             range(len(sorted_loss)//bins, len(sorted_loss), len(sorted_loss)//bins))]

    loss_pdfs = [np.histogram(losses_in_pbin, bins=len(loss_samples_per_bin[0]//10)) for losses_in_pbin in loss_samples_per_bin]
    #loss_pdfs is essentially a probability funciton for each bin that shows the likelihood of some loss value given a certain p


    test_p_indexes = np.digitize(test_ps, p_bins)
    test_loss_pdfs = [loss_pdfs[i] for i in test_p_indexes]
    test_loss_likelihoods = [pdf[0][np.digitize(test_loss, pdf[1])] for test_loss, pdf in zip(test_losses, test_loss_pdfs)]
    print(test_loss_likelihoods)
    return np.mean(test_loss_likelihoods)
