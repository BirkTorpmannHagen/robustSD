from ooddetectors import open_and_process
from vae.vae_experiment import VAEXperiment
from vae.models.vanilla_vae import ResNetVAE
from classifier.resnetclassifier import ResNetClassifier
import yaml
from domain_datasets import build_nico_dataset
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import os
from bias_samplers import *

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
import numpy as np
import seaborn as sns
import math


def get_threshold(data):
    ood = data[data["oodness"]>1]
    ind = data[data["oodness"]<=1]
    random_sampler_ind_data = ind[(ind["sampler"] == "RandomSampler")]
    if "RandomSampler" not in data["sampler"].unique():
        random_sampler_ind_data = ind[(ind["sampler"] == "ClusterSamplerWithSeverity_1.0")]
    threshold = random_sampler_ind_data["pvalue"].min()
    return threshold


def fpr(data, threshold):
    """
    :param ood_ps
    :param ind_ps
    Find p-value threshold that results in 95% TPR. Then find FPR.
    If threshold is given, use that instead.
    :return:
    """
    ood_ps = data[data["oodness"]>1]["pvalue"]

    ind_ps = data[data["oodness"]<=1]["pvalue"]
    thresholded = ind_ps<threshold
    return thresholded.mean()

def fnr(data, threshold):
    """
    :param ood_ps
    :param ind_ps
    Find p-value threshold that results in 95% TPR. Then find FPR.
    If threshold is given, use that instead.
    :return:
    """
    ood_ps = data[data["oodness"]>1]["pvalue"]

    ind_ps = data[data["oodness"]<=1]["pvalue"]
    thresholded = ood_ps>=threshold
    return thresholded.mean()
def balanced_accuracy(data, threshold):
    ood_ps = data[data["oodness"]>1]["pvalue"]
    ind_ps = data[data["oodness"]<=1]["pvalue"]
    sorted_ps = sorted(ind_ps)
    # ba = ((ind_ps>=threshold).mean()+(ood_ps<threshold).mean()) /2
    ba = 1-fpr(data, threshold) + 1-fnr(data, threshold)
    ba = ba/2
        # print(f"{(ind_ps >= threshold).mean()}+ {(ood_ps < threshold).mean()} / 2")
    return ba

def auroc(data):
    ood_ps = data[data["oodness"]>1]["pvalue"]

    ind_ps = data[data["oodness"]<=1]["pvalue"]
    true = [0]*len(ood_ps)+[1]*len(ind_ps)
    probs = list(ood_ps)+list(ind_ps)
    auc = roc_auc_score(true, probs)
    return auc

def aupr(data):
    ood_ps = data[data["oodness"]>1]["pvalue"]

    ind_ps = data[data["oodness"]<=1]["pvalue"]
    true = [0] * len(ood_ps) + [1] * len(ind_ps)
    probs = list(ood_ps) + list(ind_ps)
    auc = average_precision_score(true, probs)
    return auc

def correlation(pandas_df, plot=False, split_by_sampler=False):
    # pandas_df["pvalue"]=pandas_df["pvalue"].apply(lambda x: math.log(x, 10))
    # pandas_df = pandas_df[pandas_df["sampler"]!="ClassOrderSampler"]
    merged_p = pandas_df["pvalue"]
    merged_loss = pandas_df["loss"]

    return spearmanr(merged_p, merged_loss)[0]

def linreg_smape(pandas_df):
    pandas_df = pandas_df[pandas_df["sampler"]!="ClassOrderSampler"]
    ps = pandas_df["pvalue"].apply(lambda x: math.log10(x) if x!=0 else -250)
    sns.regplot(x=ps, y=pandas_df["loss"])
    plt.show()
    losses = pandas_df["loss"]
    lr = QuantileRegressor()
    ps_r = np.array(ps).reshape(-1, 1)
    lr.fit(ps_r, losses)
    preds = lr.predict(ps_r)
    return mape(preds, losses)


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


def risk(data, threshold):

    ood = data[data["oodness"]>1]
    ind = data[data["oodness"]<=1]

    nopredcost = 0
    # data["risk"] = data["loss"]
    # data.loc[((data["pvalue"] < threshold) & (data["oodness"] >= 1)), "risk"] = nopredcost
    # data.loc[((data["pvalue"] >= threshold) & (data["oodness"] < 1)), "risk"] = nopredcost

    data["risk"]=0
    data.loc[((data["pvalue"] < threshold) & (data["oodness"] < 1)), "risk"] = data[data["oodness"]>=1]["loss"].mean() #false positive
    data.loc[((data["pvalue"] >= threshold) & (data["oodness"] >= 1)), "risk"] = data[data["oodness"]>=1]["loss"].mean() #false negative, cost is the average loss of the ood samples

    assert -1 not in pd.unique(data["risk"]) #sanity check

    return data["risk"].mean()






def collect_losswise_metrics(fname, fnr=0.05, ood_fold_name="ood", plots=True):
    data = open_and_process(fname, filter_noise=False, combine_losses=True)

    sns.scatterplot(data=data, x="pvalue", y="loss", hue="fold")
    plt.xscale("log")
    plt.show()

    #determine sample oodness according to loss

    data["oodness"]=data["loss"]/data[data["fold"]=="ind"]["loss"].quantile(0.95)
    ood = data[data["oodness"]>1]
    ind = data[data["oodness"]<=1]

    # ood = data[(data["fold"]!="ind")]
    # ind = data[data["fold"]=="ind"]

    #find threshold for ind/ood; simulate "naive" approach of not accounting for sample bias
    threshold = get_threshold(data)
    corr = correlation(data, plot=True)

    if plots:
        fig, ax = plt.subplots(1, len(data["sampler"].unique()), figsize=(16,8), sharey=True)
    for i, sampler in enumerate(data["sampler"].unique()):
        subset = data[data["sampler"]==sampler]
        subset_ood = subset[subset["oodness"]>=1]
        subset_ind = subset[subset["oodness"]<1]
        acc =balanced_accuracy(subset, threshold)
        print(f"acc for {sampler}: {acc}")
        # print(f"fpr for {sampler}: {fpr}")
        if plots:
            ax[i].scatter(subset_ood["loss"], subset_ood["pvalue"], label="ood")
            ax[i].scatter(subset_ind["loss"], subset_ind["pvalue"], label="ind")
            ax[i].set_yscale("log")
            ax[i].hlines(threshold, 0, 1, label="threshold")
            ax[i].set_title(sampler)
            plt.legend()
    #
    if plots:
        fig.suptitle(fname)
        plt.show()

def correlation_summary():
    df = get_correlation_metrics_for_all_experiments()
    print(df.groupby(["Dataset", "Sampler",  "OOD Detector"])["Correlation"].mean())

def plot_regplots():
    # summarize overall results;
    table_data = []
    for dataset in ["CIFAR10", "CIFAR100", "NICONoise", "NjordNoise", "PolypNoise", "imagenette"]:
        for dsd in ["ks", "ks_5NN", "typicality"]:
            for sample_size in [100]:
                fname = f"data/{dataset}_{dsd}_{sample_size}_fullloss.csv"
                if dataset == "Polyp":
                    fname = f"data/{dataset}_{dsd}_{sample_size}_fullloss_ex.csv"
                    if dsd == "ks_5NN":
                        fname = f"data/{dataset}_ks_5NN_{sample_size}_fullloss_ex.csv"
                data = open_and_process(fname)
                if data is None:
                    continue
                data["Dataset"]=dataset
                data["OOD Detector"]=dsd
                table_data.append(data)
    merged = pd.concat(table_data)

    test = merged[merged["Dataset"]=="imagenette"]
    # print(test.columns)
    correlations = test.groupby(["OOD Detector", "sampler"]).apply(lambda x: correlation(x))
    g = sns.FacetGrid(data=test, col="OOD Detector", row="sampler", sharey=False, sharex=False, margin_titles=True)
    g.map_dataframe(sns.scatterplot, x="pvalue", y="loss", hue="fold",  palette="mako")
    g.set(xscale="log").set(ylim=0)
    def annotate(data, **kws):
        r = correlation(data)
        ax = plt.gca()
        ax.text(0.05, 0.8, 'r={:.2f}'.format(r), fontsize=10, fontweight="bold",
                     transform=ax.transAxes)

    g.map_dataframe(annotate)
    plt.suptitle("Imagenette", fontsize=20, y=0.97)
    plt.subplots_adjust(top=0.90)
    plt.savefig("figures/regplots_samplers.eps")
    plt.show()
    correlations = merged.groupby(["OOD Detector", "Dataset"]).apply(lambda x: correlation(x))
    g = sns.FacetGrid(data=merged, col="OOD Detector", row="Dataset", sharey=False, sharex=False, margin_titles=True)
    g.legend()
    g.map_dataframe(sns.scatterplot, x="pvalue", y="loss", hue="fold",  palette="mako")
    g.set(xscale="log").set(ylim=0)

    def annotate(data, **kws):
        r = correlation(data)
        ax = plt.gca()
        ax.set_title('r={:.2f}'.format(r), fontsize=10, fontweight="bold",
                transform=ax.transAxes)
    g.map_dataframe(annotate)
    plt.savefig("figures/regplots.eps")
    plt.show()


def illustrate_clustersampler():
    fig, ax = plt.subplots(5,1, sharex=True, sharey=True)
    for i, severity in enumerate([0, 0.1, 0.25, 0.5, 1]):
        dataset_classes = np.array(sum([[i] * 10 for i in range(10)], []))  # sorted
        shuffle_indeces = np.random.choice(np.arange(len(dataset_classes)), size=int(len(dataset_classes) * severity),
                                           replace=False)
        to_shuffle = dataset_classes[shuffle_indeces]
        np.random.shuffle(to_shuffle)
        dataset_classes[shuffle_indeces] = to_shuffle
        ax[i].imshow(dataset_classes.reshape((1,len(dataset_classes))).repeat(16,0), cmap="viridis")
        # ax[i].axis("off")
        ax[i].set_ylabel(f"{1-severity}       ", rotation=0)
        ax[i].xaxis.set_visible(False)
        # make spines (the box) invisible
        plt.setp(ax[i].spines.values(), visible=False)
        # remove ticks and labels for the left axis
        ax[i].tick_params(left=False, labelleft=False)
        # remove background patch (only needed for non-white background)
        ax[i].patch.set_visible(False)
    fig.text(0.5, 0.05, 'Index Order', ha='center')
    fig.text(0.1, 0.5, 'Bias Severity', va='center', rotation='vertical')
    plt.savefig("bias_severity.eps")
    plt.show()


def plot_bias_severity_impact(filename):
    data = pd.read_csv(filename)
    full_bias = pd.read_csv("lp_data_cifar10_noise.csv")
    full_bias = full_bias[full_bias["sampler"]=="ClusterSampler"]
    full_bias["sampler"]="ClusterSamplerWithSeverity_0.0"
    data = pd.concat((data, full_bias))

    data = data[data["sample_size"]==100]
    dataframe = []
    for ood_dataset in np.unique(data["ood_dataset"]):
        das_kn = [[], []]
        das_vn = [[], []]
        for sampler in np.unique(data["sampler"]):
            subset = data[data["sampler"]==sampler]
            ood = subset[subset["ood_dataset"] == ood_dataset]
            ind = subset[subset["ood_dataset"] == "cifar10_0.0"]
            das_kn[1].append(auroc(ood["kn_p"], ind["kn_p"]))
            das_vn[1].append(auroc(ood["vanilla_p"], ind["vanilla_p"]))
            das_vn[0].append(1-float(sampler.split("_")[-1]))
            das_kn[0].append(1 - float(sampler.split("_")[-1]))
            dataframe.append({"ood_dataset": ood_dataset, "Bias Severity":1 - float(sampler.split("_")[-1]), "AUROC": das_kn[1][-1], "Method":"KNNDSD"})
            dataframe.append({"ood_dataset": ood_dataset, "Bias Severity":1 - float(sampler.split("_")[-1]), "AUROC": das_vn[1][-1], "Method":"Rabanser et Al."})

    df = pd.DataFrame(data=dataframe)
    sns.lineplot(data=df, x="Bias Severity", y="AUROC", hue="Method")
    plt.savefig("figures/bias_severity_lineplot.png")
    plt.show()
        # plt.plot(das_kn[0], das_kn[1],label="KNNDSD")
        # plt.plot(das_vn[0], das_vn[1],label="Rabanser et Al.")
        # plt.title(str(ood_dataset))
        # plt.legend()
        # plt.show()


def breakdown_by_sample_size(placeholder=False, metric="DR"):
    df = get_classification_metrics_for_all_experiments(placeholder=placeholder)
    # print(df.groupby(["Dataset", "Sample Size"])["DR"].mean())
    # input()
    df = df.groupby(["Dataset", "OOD Detector", "Sampler", "Sample Size"])[metric].mean()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
    df = df.reset_index()
    g = sns.FacetGrid(data=df, col="Sampler", margin_titles=True)
    g.map_dataframe(sns.lineplot, x="Sample Size", y="DR", hue="OOD Detector")
    # sns.lineplot(data=df, x="Sample Size", y="DR", hue="OOD Detector")
    plt.show()

    g = sns.FacetGrid(data=df, col="Dataset", col_wrap=3, sharey=False, sharex=False)
    g.map_dataframe(sns.lineplot, x="Sample Size", y="DR", hue="OOD Detector")
    g.add_legend()
    plt.savefig("figures/samplesizebreakdown.eps")
    plt.show()

def breakdown_by_sampler(placeholder=False, metric="DR"):
    df = get_classification_metrics_for_all_experiments(placeholder=placeholder)
    df = df.groupby(["Dataset", "Sampler", "OOD Detector"])[metric].mean()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
def plot_severity(dataset,sample_size):
    df_knn = open_and_process(f"data/{dataset}_ks_5NN_{sample_size}_severity.csv", filter_noise=True)
    df_rab = open_and_process(f"data/{dataset}_ks_{sample_size}_severity.csv", filter_noise=True)
    df_typ = open_and_process(f"data/{dataset}_ks_{sample_size}_severity.csv", filter_noise=True)

    df_rab["OOD Detector"] = "Rabanser et Al."
    df_knn["OOD Detector"] = "KNNDSD"
    df_typ["OOD Detector"] = "Typicality"

    threshold_rab = get_threshold(df_rab)
    threshold_knn = get_threshold(df_knn)
    threshold_typ = get_threshold(df_typ)
    thres_dict = dict(zip(["Rabanser et Al.", "KNNDSD","Typicality"], [threshold_rab, threshold_knn, threshold_typ]))
    merged = pd.concat((df_knn, df_rab, df_typ))
    # g = sns.FacetGrid(data=merged, col="OOD Detector", row="sampler", sharey=False, sharex=False, margin_titles=True)
    # g.map_dataframe(sns.scatterplot, x="pvalue", y="loss", hue="fold",  palette="mako")
    # g.set(xscale="log").set(ylim=0)
    # plt.tight_layout()
    # plt.show()
    data = []
    # print("Sampler & KNNDSD & Rabanser\\\\")

    for sampler in merged["sampler"].unique():
        sampler_val =1-float(sampler.split('_')[-1][:4])
        # print(f"{1-float(sampler.split('_')[-1][:4]):.3}",end=": ")
        by_sampler = merged[merged["sampler"]==sampler]
        for ood_detector in by_sampler["OOD Detector"].unique():
            by_dsd = by_sampler[by_sampler["OOD Detector"]==ood_detector]
            corr = correlation(by_dsd)
            risk_val = balanced_accuracy(by_dsd, threshold=thres_dict[ood_detector])
            data.append({"Severity":sampler_val, "OOD Detector":ood_detector, "Correlation":corr, "BA": risk_val})
            # print(f"\t& {corr:.4}", end=", ")
            # print(f"\t& {risk_val:.4}", end=", ")

        # print()
    data = pd.DataFrame(data)
    # plt.ylim(0,1)
    sns.lineplot(data=data[data["OOD Detector"]=="Rabanser et Al."], x="Severity", y="BA", hue="OOD Detector")
    plt.show()
    sns.lineplot(data=data, x="Severity", y="BA", hue="OOD Detector")
    plt.show()
def summarize_results(placeholder=False):
    df = get_classification_metrics_for_all_experiments(placeholder=placeholder)
    df = df.groupby(["Dataset", "OOD Detector"])[["FPR", "FNR", "DR", "Risk"]].mean()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def threshold_plots(dataset_name, sample_size, filter_noise=False):
    data_kn = open_and_process(f"data/{dataset_name}_ks_{sample_size}_fullloss.csv", filter_noise=filter_noise)
    data_knndsd = open_and_process(f"data/{dataset_name}_ks_5NN_{sample_size}_fullloss.csv", filter_noise=filter_noise)
    data_typicality = open_and_process(f"data/{dataset_name}_typicality_{sample_size}_fullloss.csv", filter_noise=filter_noise)

    assert data_kn is not None
    assert data_knndsd is not None
    assert data_typicality is not None

    data_kn["OOD Detector"] = "Rabanser et Al."
    data_knndsd["OOD Detector"] = "KNNDSD"
    data_typicality["OOD Detector"] = "Typicality"

    data = pd.concat((data_kn, data_knndsd, data_typicality))
    g = sns.FacetGrid(data=data, col="sampler", row="OOD Detector", sharey="row", sharex=False, margin_titles=True)
    # g = sns.FacetGrid(data=data, row="sampler", col="OOD Detector", sharey=False, sharex="col", margin_titles=True)
    # print(data.head(10))
    g.map_dataframe(sns.scatterplot, x="loss", y="pvalue", hue="fold", palette="mako").set(yscale="log")
    # data["pvalue"]=data["pvalue"].apply(lambda x: np.log10(x))
    # g.map_dataframe(sns.histplot, x="pvalue", hue="fold", palette="mako", bins=20)
    plt.title(dataset_name)
    plt.savefig("figures/threshold_plots.eps")
    plt.show()

def get_correlation_metrics_for_all_experiments(placeholder=False):
    #summarize overall results;
    table_data = []
    for dataset in ["CIFAR10", "CIFAR100", "NICONoise", "NjordNoise", "PolypNoise", "imagenette"]:
        for dsd in ["ks", "ks_5NN", "typicality"]:
            for sample_size in [10, 20, 50, 100, 200, 500]:
                fname = f"data/{dataset}_{dsd}_{sample_size}_fullloss.csv"
                if dataset=="Polyp":
                    fname=f"data/{dataset}_{dsd}_{sample_size}_fullloss_ex.csv"
                    if dsd=="ks_5NN":
                        fname = f"data/{dataset}_ks_5NN_{sample_size}_fullloss_ex.csv"
                data = open_and_process(fname)
                if data is None:
                    if placeholder:
                        table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                           "FPR": -1,
                                           "DR": -1,
                                           "Risk": -1})
                    continue
                for sampler in pd.unique(data["sampler"]):
                    subset = data[data["sampler"] == sampler]
                    table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                       "Sampler": sampler,
                                       "Correlation": correlation(subset)})
    df = pd.DataFrame(data=table_data).replace("ks_5NN", "KNNDSD").replace("ks", "KS").replace(
        "typicality", "Typicality")
    return df

def get_classification_metrics_for_all_experiments(placeholder=False):
    #summarize overall results;
    table_data = []
    for dataset in ["CIFAR10", "CIFAR100", "NICO", "Njord", "Polyp", "imagenette"]:
        for dsd in ["ks", "ks_5NN", "typicality"]:
            for sample_size in [50, 100, 200, 500]:
                fname = f"data/{dataset}_{dsd}_{sample_size}_fullloss.csv"
                data = open_and_process(fname, filter_noise=True)
                if data is None:
                    if placeholder:
                        table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                           "FPR": -1,
                                           "FNR": -1,
                                           "DR": -1,
                                           "Risk": -1})
                    continue
                threshold = get_threshold(data)
                for sampler in pd.unique(data["sampler"]):
                    subset = data[data["sampler"] == sampler]
                    table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                       "Sampler": sampler,
                                       "FPR": fpr(subset, threshold=threshold),
                                       "FNR": fnr(subset, threshold=threshold),
                                       "DR": balanced_accuracy(subset, threshold=threshold),
                                       "Risk": risk(subset, threshold=threshold),
                                       "Correlation": correlation(subset)})
    df = pd.DataFrame(data=table_data).replace("ks_5NN", "KNNDSD").replace("ks", "KS").replace(
        "typicality", "Typicality")
    return df


def experiment_prediction(fname):
    data = open_and_process(fname, combine_losses=True)
    data["pvalue"]=data["pvalue"].apply(lambda x: np.log10(x))
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(axis=1, how='any', inplace=True)

    # probability approach
    num_bins = 10
    labels = range(1, num_bins + 1)  # The bin labels; ensures unique labels for each bin
    data["bin"], bin_edges = pd.qcut(data["pvalue"], q=num_bins, retbins=True, labels=labels)

    # Convert the bin labels back to integers for consistency
    data["bin"] = data["bin"].astype(int)
    # bins = np.linspace(min(data["pvalue"]), max(data["pvalue"]), num_bins)
    # data["bin"] = np.digitize(data["pvalue"], bins)
    data["bin"] = data["bin"].apply(lambda x: round(bin_edges[x-1]))
    print(data["bin"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=data, x="bin", y="loss", scatter=False)
    sns.violinplot(data=data, ax=ax, x="bin", y="loss", positions=np.unique(data["bin"]))
    plt.show()
    x = data["loss"]
    g = data["bin"]
    # df = pd.DataFrame(dict(x=x, g=g))

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(data, row="bin", hue="bin", aspect=7, height=1, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "loss",
          bw_adjust=.4, clip_on=False,
          fill=True, alpha=1, linewidth=1.5, clip=(0,None))
    g.map(sns.kdeplot, "loss", clip_on=False, color="w", lw=2, bw_adjust=.4)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=16)

    g.map(label, "loss")


    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    ylabel= g.axes[0][0].set_ylabel("log(p) bins", fontsize=16)
    ylabel.set_position((ylabel.get_position()[0], -3))
    plt.xlabel("Loss", fontsize=16)
    plt.savefig(f"figures/{fname.split('/')[-1]}_loss_vs_pvalue_pdf.eps")
    plt.show()


if __name__ == '__main__':
    """
    # Classification
    """
    # print(collect_losswise_metrics("data/Njord_ks_100_fullloss.csv"))
    # summarize_results()
    # input()
    #
    #sampler_breakdown
    # breakdown_by_sampler()
    # input()
    #
    #sample_size_breakdown
    # breakdown_by_sample_size()

    # thresholding_plots
    # threshold_plots("Njord", 100)

    #severity
    # plot_severity("imagenette", 100)
    """
    Correlation plots
    """
    correlation_summary()
    plot_regplots()
    # plot_severity("imagenette", 100)


