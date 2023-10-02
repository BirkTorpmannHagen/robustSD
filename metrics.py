import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import RocCurveDisplay
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
import numpy as np
import seaborn as sns
import math
from ast import literal_eval

def fprat95tpr(ood_ps, ind_ps, threshold=0):
    """
    :param ood_ps
    :param ind_ps
    Find p-value threshold that results in 95% TPR. Then find FPR.
    If threshold is given, use that instead.
    :return:
    """

    sorted_ps = sorted(ood_ps) #from lowest to highest
    if threshold==0:
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
    if threshold==0:
        threshold = sorted_ps[int(np.ceil(tnr_threshold*len(sorted_ps)))]
    # threshold = min(sorted_ps)

    # print("t:", threshold)

    return ((ind_ps>threshold).mean()+(ood_ps<threshold).mean()) /2

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

def correlation(pandas_df, plot=False, split_by_sampler=True):
    # pandas_df["pvalue"]=pandas_df["pvalue"].apply(lambda x: math.log(x, 10))
    pandas_df = pandas_df[pandas_df["sampler"]!="ClassOrderSampler"]
    merged_p = pandas_df["pvalue"].apply(lambda x: math.log10(x) if x!=0 else -250)
    merged_loss = pandas_df["loss"]

    if plot:
        for sampler in pd.unique(pandas_df["sampler"]):
            sns.scatterplot(data=pandas_df[pandas_df["sampler"]==sampler], x="pvalue", y="loss", label=sampler)
        # assert len(pandas_df["sampler"].unique())==1
        # plt.title(pandas_df["sampler"].unique()[0])
        plt.xscale("log")
        plt.show()
    if split_by_sampler:
        for sampler in pd.unique(pandas_df["sampler"]):
            bysampler = pandas_df[pandas_df["sampler"]==sampler]
            print(f"{sampler}: {pearsonr(bysampler['pvalue'].apply(lambda x: math.log10(x) if x!=0 else -250), bysampler['loss'])}")
    return spearmanr(merged_p, merged_loss)

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



def collect_losswise_metrics(fname, fnr=0, ood_fold_name="ood", plots=False):
    data = pd.read_csv(fname)
    #merge loss arrays
    data["loss"] = data["loss"].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])
    data["loss"] = data["loss"].apply(lambda x: np.mean(x))
    # data= data.explode("loss")
    # print(data)
    # input()
    #determine sample oodness according to loss
    if ood_fold_name!="ood":
        data = data[(data["fold"]=="ind")|(data["fold"]==ood_fold_name)]
    data["oodness"]=data["loss"]/data[data["fold"]=="ind"]["loss"].quantile(0.95)
    ood = data[data["oodness"]>=1]
    ind = data[data["oodness"]<1]

    # ood = data[(data["fold"]!="ind")]
    # ind = data[data["fold"]=="ind"]

    if plots:
        ax = sns.scatterplot(x="loss", y="pvalue", hue="fold", data=data)
        plt.yscale("log")
        plt.show()


    #find threshold for ind/ood; simulate "naive" approach of not accounting for sample bias
    random_sampler_ind_data = ind[(ind["sampler"]=="RandomSampler")]
    sorted_ind_ps = sorted(random_sampler_ind_data["pvalue"])
    threshold = sorted_ind_ps[int(np.ceil(fnr*len(sorted_ind_ps)))] # min p_value for a sample to be considered ind
    corr = correlation(data, plot=True)
    print(corr)
    if plots:
        fig, ax = plt.subplots(1, len(data["sampler"].unique()), figsize=(16,8), sharey=True)
    for i, sampler in enumerate(data["sampler"].unique()):
        subset_ind = ind[ind["sampler"]==sampler]
        subset_ood = ood[ood["sampler"]==sampler]
        merged = pd.concat([subset_ind, subset_ood])
        acc = calibrated_detection_rate(subset_ood["pvalue"], subset_ind["pvalue"], threshold=threshold)
        fpr = fprat95tpr(subset_ood["pvalue"], subset_ind["pvalue"], threshold=threshold)

        # print(f"acc for {sampler}: {acc}")
        # print(f"fpr for {sampler}: {fpr}")
        if plots:
            ax[i].scatter(subset_ood["loss"], subset_ood["pvalue"], label="ood")
            ax[i].scatter(subset_ind["loss"], subset_ind["pvalue"], label="ind")
            ax[i].set_yscale("log")
            ax[i].hlines(threshold, 0, 10, label="threshold")
            # ax[i].set_ylim((1e-7, 0))
            ax[i].set_title(sampler)
            plt.legend()
    #
    if plots:
        fig.suptitle(fname)
        plt.show()


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