import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import RocCurveDisplay
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
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

def calibrated_detection_rate(ood_ps, ind_ps, tnr_threshold=0, threshold=0):
    """

    :param ood_ps:
    :param ind_ps:

    Essentially, fnr&0fpr
    :return:
    """
    sorted_ps = sorted(ind_ps)
    threshold = sorted_ps[int(np.ceil(tnr_threshold*len(sorted_ps)))]
    # threshold = min(sorted_ps)

    # print("t:", threshold)

    return ((1-tnr_threshold)+(ood_ps<threshold).mean()) /2

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

def linreg_smape(ps, loss, ood_ps, ood_loss):
    lr = LinearRegression()
    lr.fit(np.array(np.log10(ps)).reshape(-1, 1), loss)
    preds = lr.predict(np.array(np.log10(ood_ps)).reshape(-1, 1))
    return mape(preds, ood_loss)


def get_loss_pdf_from_ps(ps, loss, test_ps, test_losses, bins=15):
    """
        Computes a pdf for the given number of bins, and gets the likelihood of the test loss at the given test_ps bin.
        #todo: collect a new noise dataset with the right predictor
        :returns the average likelihood of the observed test-loss as bootstrapped from the pdf w/noise.
        Higher likelihood ~ more likely that the model is correct more often.
        
    """
    #split ps into unevenly sized bins with equal number of entries
    pargsort = np.argsort(ps)
    sorted_ps = np.array(ps)[pargsort]
    sorted_loss = np.array(loss)[pargsort]
    p_bins = sorted_ps[::len(sorted_ps)//bins] #defines bin limits
    [min_val, max_val] = [sorted_ps[0], sorted_ps[-1]]
    p_bins = np.append(p_bins, max_val)

    #there are now 15 bins
    # print(p_bins)
    loss_samples_per_bin = [sorted_loss[i:j] for i, j in zip(range(0, len(sorted_loss), len(sorted_loss)//bins),
                                                             range(len(sorted_loss)//bins, len(sorted_loss)+len(sorted_loss)//bins, len(sorted_loss)//bins))]

    loss_pdfs = [np.histogram(losses_in_pbin, bins=len(loss_samples_per_bin[0])//10) for losses_in_pbin in loss_samples_per_bin]
    #loss_pdfs is essentially a probability funciton for each bin that shows the likelihood of some loss value given a certain p

    test_p_indexes = np.digitize(test_ps, p_bins[:-1])
    loss_diff = []
    for p, loss in zip(test_ps, test_losses):
        index = np.digitize(p, p_bins[:-1])
        predicted_loss = np.mean(loss_samples_per_bin[np.clip(index, 0, len(loss_samples_per_bin)-1)])
        loss_diff.append(np.abs(loss-predicted_loss))
    return np.mean(loss_diff)
    #index of the p-bin that each test point falls into

    # print(np.max(test_p_indexes))
    # print(len(loss_pdfs))
    # test_loss_pdfs = [loss_pdfs[np.clip(i, 0,len(loss_pdfs)-1)] for i in test_p_indexes]
    # test_loss_likelihoods = [pdf[0][np.clip(np.digitize(test_loss, pdf[1]), 0, 3)] for test_loss, pdf in zip(test_losses, test_loss_pdfs)]
    # print(test_loss_likelihoods)
    # return np.mean(test_loss_likelihoods)


def risk(fname, fnr=0):
    data = pd.read_csv(fname)

    plt.hist(data[data["fold"]=="ind"]["loss"], alpha=0.5, label="ind")
    plt.hist(data[data["fold"] != "ind"]["loss"], alpha=0.5, label="ood")
    plt.legend()
    plt.show()

    print(data[data["fold"]=="ind"]["loss"].max())
    print(data[data["fold"]=="ind"]["loss"].mean())
    print(data[data["fold"]!="ind"]["loss"].max())
    print(data[data["fold"]!="ind"]["loss"].mean())
    data["oodness"]=data["loss"]/data[data["fold"]=="ind"]["loss"].max()
    print(data["oodness"])
    random_sampler_ind_data = data[(data["sampler"]=="RandomSampler")&(data["fold"]=="ind")]
    sorted_ind_ps = sorted(random_sampler_ind_data["pvalue"])
    threshold = sorted_ind_ps[int(np.ceil(fnr*len(sorted_ind_ps)))] # min p_value for a sample to be considered ind

    ind_data = data[data["fold"]=="ind"]
    ood_data = data[(data["fold"]!="ind") & data["fold"]!="dim"]


    total_risk = 0
    for sampler in data["sampler"].unique():
        subset = data[data["sampler"]==sampler]
        subset_ind = subset[subset["fold"]=="ind"]
        subset_ood = subset[(subset["fold"] != "ind") & (subset["fold"] != "dim")]
        # risk is avg ood loss if false positive, relevant ood loss if false negative


        #tp risk + fp risk + tn risk + fn risk

        risk_val = ((subset_ind["pvalue"]<threshold)*subset_ood["loss"].mean()).mean()\
                   + ((subset_ood["pvalue"]>threshold)*subset_ood["loss"]).mean()
        print(f"{sampler} risk: {risk_val}")

def risk_across_noises(fname, fnr=0):
    data_full = pd.read_csv(fname)
    for fold in data_full["fold"].unique():
        if fold=="ind":
            continue
        data = data_full[(data_full["fold"]==fold) | (data_full["fold"]=="ind")]
        print(np.unique(data["fold"]))

        random_sampler_ind_data = data[(data["sampler"] == "RandomSampler") & (data["fold"] == "ind")]
        sorted_ind_ps = sorted(random_sampler_ind_data["pvalue"])
        threshold = sorted_ind_ps[int(np.ceil(fnr * len(sorted_ind_ps)))]  # min p_value for a sample to be considered ind

        ind_data = data[data["fold"] == "ind"]
        ood_data = data[(data["fold"] != "ind") & data["fold"] != "dim"]
        total_risk = 0
        for sampler in data["sampler"].unique():
            subset = data[data["sampler"] == sampler]
            subset_ind = subset[subset["fold"] == "ind"]
            subset_ood = subset[(subset["fold"] != "ind") & (subset["fold"] != "dim")]
            print(f"{fold}, {sampler}: ", calibrated_detection_rate(subset_ood["pvalue"], subset_ind["pvalue"]))
            input()
            # risk is avg ood loss if false positive, relevant ood loss if false negative

            # tp risk + fp risk + tn risk + fn risk

            risk_val = ((subset_ind["pvalue"] < threshold) * subset_ood["loss"].mean()).mean() \
                       + ((subset_ood["pvalue"] > threshold) * subset_ood["loss"]).mean()
            print(f"{sampler} at {fold} risk: {risk_val}")