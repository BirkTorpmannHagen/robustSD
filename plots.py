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


def get_threshold(data, fpr=0):
    ood = data[data["oodness"]>=1]
    ind = data[data["oodness"]<1]
    random_sampler_ind_data = ind[(ind["sampler"] == "RandomSampler")]
    sorted_ind_ps = sorted(random_sampler_ind_data["pvalue"])
    threshold = sorted_ind_ps[0]
    return threshold


def fpr(data, threshold=0):
    """
    :param ood_ps
    :param ind_ps
    Find p-value threshold that results in 95% TPR. Then find FPR.
    If threshold is given, use that instead.
    :return:
    """
    ood_ps = data[data["oodness"]>=1]["pvalue"]
    ind_ps = data[data["oodness"]<1]["pvalue"]
    thresholded = ind_ps<threshold
    return thresholded.mean()

def calibrated_detection_rate(data, threshold):
    ood_ps = data[data["oodness"]>=1]["pvalue"]
    ind_ps = data[data["oodness"]<1]["pvalue"]
    sorted_ps = sorted(ind_ps)
    return ((ind_ps>=threshold).mean()+(ood_ps<threshold).mean()) /2

def auroc(data):
    ood_ps = data[data["oodness"]>=1]["pvalue"]
    ind_ps = data[data["oodness"]<1]["pvalue"]
    true = [0]*len(ood_ps)+[1]*len(ind_ps)
    probs = list(ood_ps)+list(ind_ps)
    auc = roc_auc_score(true, probs)
    return auc

def aupr(data):
    ood_ps = data[data["oodness"]>=1]["pvalue"]
    ind_ps = data[data["oodness"]<1]["pvalue"]
    true = [0] * len(ood_ps) + [1] * len(ind_ps)
    probs = list(ood_ps) + list(ind_ps)
    auc = average_precision_score(true, probs)
    return auc

def correlation(pandas_df, plot=False, split_by_sampler=False):
    # pandas_df["pvalue"]=pandas_df["pvalue"].apply(lambda x: math.log(x, 10))
    # pandas_df = pandas_df[pandas_df["sampler"]!="ClassOrderSampler"]
    merged_p = pandas_df["pvalue"].apply(lambda x: math.log10(x) if x!=0 else -250)
    merged_loss = pandas_df["loss"]

    if plot:
        for sampler in pd.unique(pandas_df["sampler"]):
            sns.scatterplot(data=pandas_df[pandas_df["sampler"]==sampler], x="pvalue", y="loss", label=sampler)
        plt.xscale("log")
        plt.show()
    if split_by_sampler:
        for sampler in pd.unique(pandas_df["sampler"]):
            bysampler = pandas_df[pandas_df["sampler"]==sampler]
            print(f"{sampler}: {pearsonr(bysampler['pvalue'].apply(lambda x: math.log10(x) if x!=0 else -250), bysampler['loss'])}")
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

    ood = data[data["oodness"]>=1]
    ind = data[data["oodness"]<1]
    # fp = len(ind[ind["pvalue"]<threshold])
    # tp = len(ood[ood["pvalue"]<threshold])
    # fn = len(ind[ind["pvalue"]>=threshold])
    # tn = len(ood[ood["pvalue"]>=threshold])
    #
    # assert fp+tp+fn+tn == len(data)
    # # tncosts = ind["loss"].median() #cost of predictive model not predicting
    # tncosts = 0
    # fpcost = ood["loss"].median() # cost of predictive model not predicting when it should have, as bad as predicting wrong
    # fncost = ood["loss"].median() # cost of predictive model predicting when it shouldn't have; ie predicting incorrectly
    # tpcost = 0
    # # tpcost = ind["loss"].median() # cost of predictive model predicting correctly
    # risk = (fp*fpcost+tp*tpcost+fn*fncost+tn*tncosts)/len(data)


    nopredcost = ind["loss"].median()
    data["risk"]=0
    data.loc[((data["pvalue"] < threshold) & (data["oodness"] >= 1)), "risk"] = nopredcost #true positive
    data.loc[((data["pvalue"] < threshold) & (data["oodness"] < 1)), "risk"] = data[data["oodness"]>=1]["loss"].mean() #false positive
    data.loc[((data["pvalue"] >= threshold) & (data["oodness"] >= 1))] = data[data["oodness"]>=1]["loss"].mean() #false negative, cost is the average loss of the ood samples
    data.loc[((data["pvalue"] >= threshold) & (data["oodness"] < 1))] = nopredcost #true negative, cost is the average loss of the ind samples
    assert 0 not in pd.unique(data["risk"]) #sanity check
    data.loc[data["pvalue"] >= threshold, "risk"] = data.loc[data["pvalue"] >= threshold, "loss"] #true negative, cost

    # for sampler in pd.unique(data["sampler"]):
    #     bysampler = data[data["sampler"]==sampler]
    #     print(f"{sampler}: {bysampler['risk'].mean()}")

    return data["risk"].mean()






def collect_losswise_metrics(fname, fnr=0.05, ood_fold_name="ood", plots=True):
    data = open_and_process(fname, filter_noise=False, combine_losses=True)
    sns.scatterplot(data=data, x="pvalue", y="loss", hue="fold")
    plt.xscale("log")
    plt.show()

    #determine sample oodness according to loss
    if ood_fold_name!="ood":
        data = data[(data["fold"]=="ind")|(data["fold"]==ood_fold_name)]
    data["oodness"]=data["loss"]/data[data["fold"]=="ind"]["loss"].quantile(0.95)
    ood = data[data["oodness"]>=1]
    ind = data[data["oodness"]<1]

    # ood = data[(data["fold"]!="ind")]
    # ind = data[data["fold"]=="ind"]

    #find threshold for ind/ood; simulate "naive" approach of not accounting for sample bias
    random_sampler_ind_data = ind[(ind["sampler"]=="RandomSampler")]
    sorted_ind_ps = sorted(random_sampler_ind_data["pvalue"])
    threshold = sorted_ind_ps[int(np.ceil(fnr*len(sorted_ind_ps)))] # min p_value for a sample to be considered ind
    corr = correlation(data, plot=True)

    if plots:
        fig, ax = plt.subplots(1, len(data["sampler"].unique()), figsize=(16,8), sharey=True)
    for i, sampler in enumerate(data["sampler"].unique()):
        subset = data[data["sampler"]==sampler]
        subset_ood = subset[subset["oodness"]>=1]
        subset_ind = subset[subset["oodness"]<1]
        acc =calibrated_detection_rate(subset, threshold)
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


def plot_nico_class_bias():
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))

    model = ResNetClassifier.load_from_checkpoint("/home/birk/Projects/robustSD/lightning_logs/version_0/checkpoints/epoch=109-step=1236510.ckpt", num_classes=num_classes, resnet_version=34).to("cuda")
    pca = PCA(n_components=2)
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((512, 512)),
                                transforms.ToTensor(), ])
    ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
    ind_encodings = np.zeros((len(ind), model.latent_dim))
    val_encodings = np.zeros((len(ind_val), model.latent_dim))
    ind_classes = np.zeros((len(ind), 1))
    val_classes = np.zeros((len(ind_val), 1))

    with torch.no_grad():
        for i, (x, y, _) in tqdm(enumerate(DataLoader(ind, shuffle=True)), total=len(ind)):
            ind_encodings[i] = model.encode(x.to("cuda"))[0].cpu().numpy()
            ind_classes[i] = y

        for i, (x, y, _) in tqdm(enumerate(DataLoader(ind_val, sampler=ClassOrderSampler(ind_val))),
                                 total=len(ind_val)):
            val_encodings[i] = model.encode(x.to("cuda"))[0].cpu().numpy()
            val_classes[i] = y

    pca.fit(np.vstack((ind_encodings, val_encodings)))
    transformed_ind = pca.transform(ind_encodings)
    transformed_val = pca.transform(val_encodings)
    plt.show()
    sample_size = 200
    ps_biased = []
    ps_unbiased = []
    for idx, (i, j) in enumerate(zip(np.arange(0, len(ind_val), sample_size)[:-1],
                                     np.arange(0, len(ind_val), sample_size)[1:])):
        i = int(i)
        j = int(j)
        plt.scatter(transformed_val[:, 0], transformed_val[:, 1],c="orange", label=f"InD")
        plt.scatter(transformed_val[i:j, 0],
                    transformed_val[i:j, 1], c=val_classes[i:j],cmap="tab10", label=f"Biased")
        plt.legend()
        rand = np.random.randint(0, len(transformed_val) - sample_size)
        ps_unbiased.append(ks_2samp(transformed_ind[:, 1], transformed_val[rand: rand+sample_size, 1])[1])
        ps_biased.append(ks_2samp(transformed_ind[:, 0], transformed_val[i:j, 0])[1])
        plt.title(f"{i}-{j}: {ks_2samp(transformed_ind[:, 0], transformed_val[i:j, 0])[1]} vs {ks_2samp(transformed_ind[:, 1], transformed_val[rand: rand+sample_size, 1])[1]}")
        plt.savefig(f"figures/nico_clusterbias_{idx}.eps")
        plt.show()
    print(f"Biased: {np.mean(ps_biased)}")
    print(f"Unbiased: {np.mean(ps_unbiased)}")
def plot_nico_clustering_bias():
    config = yaml.safe_load(open("vae/configs/vae.yaml"))
    model = ResNetVAE().to("cuda").eval()
    vae_exp = VAEXperiment(model, config)
    vae_exp.load_state_dict(
        torch.load("vae_logs/nico_dim/version_40/checkpoints/last.ckpt")[
            "state_dict"])
    model.eval()
    pca = PCA(n_components=2)
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
    ind_encodings = np.zeros((len(ind), model.latent_dim))
    val_encodings = np.zeros((len(ind_val), model.latent_dim))
    with torch.no_grad():
        for i, (x,y,_) in tqdm(enumerate(DataLoader(ind, shuffle=True)), total=len(ind)):
            ind_encodings[i]= model.encode(x.to("cuda"))[0].cpu().numpy()

        for i, (x,y, _) in tqdm(enumerate(DataLoader(ind_val, sampler=ClusterSampler(ind_val, model))), total=len(ind_val)):
            val_encodings[i] = model.encode(x.to("cuda"))[0].cpu().numpy()

    pca.fit(ind_encodings)
    transformed_ind = pca.transform(ind_encodings)
    transformed_val = pca.transform(val_encodings)

    for idx, (i,j) in enumerate(zip(np.linspace(0,int(.9*len(transformed_val)),100), np.linspace(int(0.1*len(transformed_val)), len(transformed_val), 100))):
        i = int(i)
        j = int(j)
        plt.scatter(transformed_val[:,0], transformed_val[:,1], label=f"InD")
        plt.scatter(transformed_val[i:j, 0],
                    transformed_val[i:j, 1], label=f"Biased")
        plt.legend()
        plt.savefig(f"figures/nico_clusterbias_{idx}.eps")
        plt.show()



def get_nico_classification_metrics(filename):
    data = pd.read_csv(filename)
    for sampler in np.unique(data["sampler"]):
        dataset = data[data["sampler"]==sampler]
        print(sampler)
        for sample_size in np.unique(dataset["sample_size"]):
            subset = dataset[dataset["sample_size"] == sample_size]
            ood = subset[subset["ood_dataset"] != "nico_dim"]
            ind = subset[subset["ood_dataset"] == "nico_dim"]
            print("sample size ",sample_size)
            fpr_van = fpr(ood["vanilla_p"], ind["vanilla_p"])
            fpr_kn = fpr(ood["kn_p"], ind["kn_p"])
            print("vanilla FPR: ", fpr_van)
            print("kn FPR:", fpr_kn)
            aupr_van = aupr(ood["vanilla_p"], ind["vanilla_p"])
            aupr_kn = aupr(ood["kn_p"], ind["kn_p"])
            print("vanilla AUPR: ", aupr_van)
            print("kn AUPR:", aupr_kn)
            auroc_van = auroc(ood["vanilla_p"], ind["vanilla_p"])
            auroc_kn = auroc(ood["kn_p"], ind["kn_p"])
            print("vanilla AUROC: ", auroc_van)
            print("kn AUROC:", auroc_kn)
            dr_van = calibrated_detection_rate(ood["vanilla_p"], ind["vanilla_p"])
            dr_kn = calibrated_detection_rate(ood["kn_p"], ind["kn_p"])
            print("vanilla DR: ", dr_van)
            print("kn DR: ", dr_kn)
            print()

def get_polyp_classification_metrics(filename):
    data = pd.read_csv(filename)
    for sampler in np.unique(data["sampler"]):
        dataset = data[data["sampler"]==sampler]
        print()
        print(sampler)
        for sample_size in np.unique(dataset["sample_size"]):
            print(sample_size)
            subset = dataset[dataset["sample_size"] == sample_size]
            ood = subset[subset["ood_dataset"] == "polyp_ood"]
            ind = subset[subset["ood_dataset"] == "polyp_ind"]
            fpr_van = fpr(ood["vanilla_p"], ind["vanilla_p"])
            fpr_kn = fpr(ood["kn_p"], ind["kn_p"])
            print("vanilla FPR: ", fpr_van)
            print("kn FPR:", fpr_kn)
            aupr_van = aupr(ood["vanilla_p"], ind["vanilla_p"])
            aupr_kn = aupr(ood["kn_p"], ind["kn_p"])
            print("vanilla AUPR: ", aupr_van)
            print("kn AUPR:", aupr_kn)
            auroc_van = auroc(ood["vanilla_p"], ind["vanilla_p"])
            auroc_kn = auroc(ood["kn_p"], ind["kn_p"])
            print("vanilla AUROC: ", auroc_van)
            print("kn AUROC:", auroc_kn)
            dr_van = calibrated_detection_rate(ood["vanilla_p"], ind["vanilla_p"])
            dr_kn = calibrated_detection_rate(ood["kn_p"], ind["kn_p"])
            print("vanilla DR: ", dr_van)
            print("kn DR: ", dr_kn)
        print()

def get_cifar10_classification_metrics(filename):
    data = pd.read_csv(filename)
    for sampler in np.unique(data["sampler"]):
        dataset = data[data["sampler"]==sampler]
        print()
        print(sampler)
        for sample_size in np.unique(dataset["sample_size"]):
            print(sample_size)
            subset = dataset[dataset["sample_size"] == sample_size]
            for noise_val in np.unique(subset["ood_dataset"]):
                if noise_val=="cifar10_0.0":
                    continue
                ood = subset[subset["ood_dataset"] == noise_val]
                ind = subset[subset["ood_dataset"] == "cifar10_0.0"]
                print(f"{noise_val} has loss {ood['loss'].mean()} compared to {ind['loss'].mean()}")
                fpr_van = fpr(ood["vanilla_p"], ind["vanilla_p"])
                fpr_kn = fpr(ood["kn_p"], ind["kn_p"])
                print("vanilla FPR: ", fpr_van)
                print("kn FPR:", fpr_kn)
                aupr_van = aupr(ood["vanilla_p"], ind["vanilla_p"])
                aupr_kn = aupr(ood["kn_p"], ind["kn_p"])
                print("vanilla AUPR: ", aupr_van)
                print("kn AUPR:", aupr_kn)
                auroc_van = auroc(ood["vanilla_p"], ind["vanilla_p"])
                auroc_kn = auroc(ood["kn_p"], ind["kn_p"])
                print("vanilla AUROC: ", auroc_van)
                print("kn AUROC:", auroc_kn)
                dr_van = calibrated_detection_rate(ood["vanilla_p"], ind["vanilla_p"])
                dr_kn = calibrated_detection_rate(ood["kn_p"], ind["kn_p"])
                print("vanilla DR: ", dr_van)
                print("kn DR: ", dr_kn)
        print()



def get_corrrelation_metrics(filename_noise, filename_ood=""):
    data = pd.read_csv(filename_noise)
    print(data.groupby(["sampler", "sample_size", "ood_dataset"])["loss"].mean())
    for sampler in np.unique(data["sampler"]):
        dataset = data[data["sampler"] == sampler]
        print()
        print(sampler)
        for sample_size in np.unique(dataset["sample_size"]):
            print(sample_size)
            subset = dataset[dataset["sample_size"] == sample_size]
            corr_van = correlation(np.log10(subset["vanilla_p"]), subset["loss"])
            corr_kn = correlation(np.log10(subset["kn_p"]), subset["loss"])
            print(sample_size)
            print("correlation vanilla", corr_van)
            print("correlation kn", corr_kn)
            f, ax = plt.subplots(figsize=(7, 7))
            # sns.regplot(np.log10(subset["kn_p"]),subset["loss"], ax=ax, color="blue")
            sns.scatterplot(np.log10(subset["kn_p"]),subset["loss"],hue=subset["ood_dataset"])

            ax.set_xlabel("kn_p", color="blue", fontsize=14)
            # ax.set(xscale="log")
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(np.array(np.log10(subset["kn_p"])).reshape(-1,1),np.array(subset["loss"]).reshape(-1,1))
            # sns.regplot(subset["vanilla_p"], subset["loss"],  ax=ax2, color="orange")
            plt.ylim((-1,np.max(subset["loss"])))
            plt.legend()
            plt.title(f"n={sample_size} and {sampler}")

            plt.savefig(f"correlation_figures/cifar10_kn_{sample_size}_{sampler}.eps")
            plt.show()

            # ax2.set(xscale="log")
            # sns.regplot(np.log(subset["vanilla_p"]), subset["loss"],  ax=ax2, color="orange")
            sns.scatterplot(np.log(subset["vanilla_p"]), subset["loss"], hue=subset["ood_dataset"])

            plt.ylim((0,np.max(subset["loss"])))
            lr = LinearRegression()
            lr.fit(np.array(np.log10(subset[subset["vanilla_p"]!=0]["vanilla_p"])).reshape(-1,1),np.array(subset[subset["vanilla_p"]!=0]["loss"]).reshape(-1,1))

            plt.title(f"n={sample_size} and {sampler}")
            plt.savefig(f"correlation_figures/cifar10_vanilla_{sample_size}_{sampler}.eps")
            plt.show()
        # predictive
        if filename_ood!="":
            dataset_ood = pd.read_csv(filename_ood)
            for sample_size in np.unique(dataset["sample_size"]):
                subset = dataset[dataset["sample_size"] == sample_size]
                vanilla_predictive_likelihood = get_loss_pdf_from_ps(subset["vanilla_p"], subset["loss"], dataset_ood["vanilla_p"], dataset_ood["loss"])
                kn_predictive_likelihood = get_loss_pdf_from_ps(subset["kn_p"], subset["loss"], dataset_ood["kn_p"], dataset_ood["loss"])
                print("vanilla predictive likelihood", vanilla_predictive_likelihood)
                print("kn predictive likelihood", kn_predictive_likelihood)
                kn_mape = linreg_smape(subset["kn_p"], subset["loss"], dataset_ood["kn_p"], dataset_ood["loss"])
                vanilla_mape = linreg_smape(subset["vanilla_p"], subset["loss"], dataset_ood["vanilla_p"], dataset_ood["loss"])
                print("vanilla mape: ", vanilla_mape)
                print("kn mape: ", kn_mape)

def genfailure_metrics(filename):
    dataset = pd.read_csv(filename)
    print(dataset.groupby(["ood_dataset"]).mean()["loss"])


def plot_loss_v_encodings():
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    config = {"device":"cuda"}

    model = ResNetClassifier.load_from_checkpoint(
        "/home/birk/Projects/robustSD/lightning_logs/version_0/checkpoints/epoch=109-step=1236510.ckpt",
        num_classes=num_classes, resnet_version=34).to("cuda")
    pca = PCA(n_components=2)
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((512, 512)),
                                transforms.ToTensor(), ])
    # ind_dataset, ood_dataset = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
    ind_dataset, ood_dataset = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="rock", seed=0)

    ood_latents = np.zeros((len(ood_dataset), model.latent_dim))
    ood_losses = np.zeros(len(ood_dataset))
    for i, (x, y, _) in tqdm(enumerate(DataLoader(ood_dataset)), total=len(ood_dataset)):
        with torch.no_grad():
            ood_latents[i] = model.encode(x.to(config["device"]))[0].cpu().numpy()

            ood_losses[i] = model.compute_loss(x.to(config["device"]),
                                                         y.to(config["device"])).cpu().numpy()
    latents =pca.fit_transform(ood_latents)
    plt.show()
    plt.hist(ood_losses, bins=100)
    plt.show()
    plt.scatter(latents[:,0], ood_losses)

    plt.show()
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(latents[:,0], latents[:,1], ood_losses, c=ood_losses)
    plt.show()

def eval_sample_size_impact(filename):
    dataset = pd.read_csv(filename)
    fig, ax = plt.subplots(3,1, sharex=True)
    for i, sampler in enumerate(np.unique([dataset["sampler"]])):
        data = dataset[dataset["sampler"] == sampler]
        dataframe = []
        for ood_dataset in np.unique(data["ood_dataset"]):
            if ood_dataset=="cifar10_0.0":
                continue
            das_kn = [[], []]
            das_vn = [[], []]
            for sample_size in np.unique(data["sample_size"]):
                subset = data[data["sample_size"] == sample_size]
                ood = subset[subset["ood_dataset"] == ood_dataset]
                ind = subset[subset["ood_dataset"] == "cifar10_0.0"]
                das_kn[1].append(auroc(ood["kn_p"], ind["kn_p"]))
                das_vn[1].append(auroc(ood["vanilla_p"], ind["vanilla_p"]))
                das_vn[0].append(sample_size)
                das_kn[0].append(sample_size)
                dataframe.append(
                    {"ood_dataset": ood_dataset, "Sample Size": sample_size, "AUROC": das_kn[1][-1],
                     "Method": "KNNDSD"})
                dataframe.append(
                    {"ood_dataset": ood_dataset, "Sample Size": sample_size, "AUROC": das_vn[1][-1],
                     "Method": "Rabanser et Al."})

        df = pd.DataFrame(data=dataframe)
        print(df)
        sns.lineplot(data=df, ax=ax[i], x="Sample Size", y="AUROC", hue="Method")
        ax[i].set_ylabel(f"{sampler}")
    ax[0].legend("")
    ax[1].legend("")
    fig.text(0.04, 0.5, "AUROC", va='center', rotation='vertical')
    plt.savefig("figures/sample_size_lineplot.png")
    plt.show()

def eval_k_impact(filename):
    #todo collect a csv with k-data for some dataset and sample size
    das_kn = []
    das_vn = []
    dataset = pd.read_csv(filename)
    for k in np.unique(dataset["k"]):
        print(k)
        subset = dataset[dataset["k"] == k]
        ood = subset[subset["ood_dataset"] != "cifar10_0.0"]
        ind = subset[subset["ood_dataset"] == "cifar10_0.0"]
        das_kn.append(calibrated_detection_rate(ood["kn_p"], ind["kn_p"]))
        das_vn.append(calibrated_detection_rate(ood["vanilla_p"], ind["vanilla_p"]))
    print(das_kn)
    plt.plot(np.unique(dataset["k"]), das_kn, label="kn")
    plt.plot(np.unique(dataset["k"]), das_vn, label="vanilla")
    plt.legend()
    plt.show()

def get_njord_classification_metrics(filename):
    data = pd.read_csv(filename)
    for sampler in np.unique(data["sampler"]):
        dataset = data[data["sampler"] == sampler]
        print(sampler)
        for sample_size in np.unique(dataset["sample_size"]):
            print(sample_size)
            subset = dataset[dataset["sample_size"] == sample_size]
            print(subset[subset["ood_dataset"] == "Njordind"]["vanilla_p"].mean()-subset[subset["ood_dataset"] == "Njordood"]["vanilla_p"].mean(), subset[subset["ood_dataset"] == "Njordind"]["kn_p"].mean()-subset[subset["ood_dataset"] == "Njordood"]["kn_p"].mean())
            ood = subset[subset["ood_dataset"] == "Njordood"]
            ind = subset[subset["ood_dataset"] == "Njordind"]
            if sample_size==100:
                subset.to_csv("debug.csv")
                input()
            fpr_van = fpr(ood["vanilla_p"], ind["vanilla_p"])
            fpr_kn = fpr(ood["kn_p"], ind["kn_p"])
            print("vanilla FPR: ", fpr_van)
            print("kn FPR:", fpr_kn)
            aupr_van = aupr(ood["vanilla_p"], ind["vanilla_p"])
            aupr_kn = aupr(ood["kn_p"], ind["kn_p"])
            print("vanilla AUPR: ", aupr_van)
            print("kn AUPR:", aupr_kn)
            auroc_van = auroc(ood["vanilla_p"], ind["vanilla_p"])
            auroc_kn = auroc(ood["kn_p"], ind["kn_p"])
            print("vanilla AUROC: ", auroc_van)
            print("kn AUROC:", auroc_kn)
            dr_van = calibrated_detection_rate(ood["vanilla_p"], ind["vanilla_p"])
            dr_kn = calibrated_detection_rate(ood["kn_p"], ind["kn_p"])
            print("vanilla DR: ", dr_van)
            print("kn DR: ", dr_kn)
        print()

def plot_nico_samplesize(filename):
    data = pd.read_csv(filename)
    for sampler in np.unique(data["sampler"]):
        dataset = data[data["sampler"]==sampler]
        auprs_kn = []
        auprs_van = []
        for sample_size in np.unique(dataset["sample_size"]):
            subset = dataset[dataset["sample_size"] == sample_size]
            ood = subset[subset["ood_dataset"] != "nico_dim"]
            ind = subset[subset["ood_dataset"] == "nico_dim"]
            auprs_van.append(aupr(ood["vanilla_p"], ind["vanilla_p"]))
            auprs_kn.append(aupr(ood["kn_p"], ind["kn_p"]))
        plt.plot(sorted(np.unique(dataset["sample_size"])), auprs_kn, label="knndsd")
        plt.plot(sorted(np.unique(dataset["sample_size"])), auprs_van, label="Rabanser et Al")
        plt.title(type(sampler).__name__)
        plt.legend()
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
            print(sampler)
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
    print(df)
    sns.lineplot(data=df, x="Bias Severity", y="AUROC", hue="Method")
    plt.savefig("figures/bias_severity_lineplot.png")
    plt.show()
        # plt.plot(das_kn[0], das_kn[1],label="KNNDSD")
        # plt.plot(das_vn[0], das_vn[1],label="Rabanser et Al.")
        # plt.title(str(ood_dataset))
        # plt.legend()
        # plt.show()


def breakdown_by_sample_size(placeholder=False, metric="DR"):
    df = get_metrics_for_all_experiments(placeholder=placeholder)
    df = df.groupby([ "Dataset", "Sample Size", "OOD Detector"])[metric].mean()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
    df = df.reset_index()
    sns.lineplot(data=df, x="Sample Size", y="DR", hue="OOD Detector")
    plt.show()

    g = sns.FacetGrid(data=df, col="Dataset", col_wrap=3, sharey=False, sharex=False)
    g.map_dataframe(sns.lineplot, x="Sample Size", y="DR", hue="OOD Detector")
    g.add_legend()
    plt.show()

def breakdown_by_sampler(placeholder=False, metric="DR"):
    df = get_metrics_for_all_experiments(placeholder=placeholder)
    df = df.groupby(["Dataset", "Sampler", "OOD Detector"])[metric].mean()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def get_metrics_for_all_experiments(placeholder=False):
    #summarize overall results;
    table_data = []
    for dataset in ["CIFAR10", "CIFAR100", "NICO", "Njord", "Polyp", "imagenette"]:
        for dsd in ["ks", "ks_5NN", "typicality"]:
            for sample_size in [10, 20, 50, 100, 200, 500]:
                fname = f"data/{dataset}_{dsd}_{sample_size}_fullloss.csv"
                data = open_and_process(fname)
                if data is None:
                    if placeholder:
                        table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                           "FPR": -1,
                                           "DR": -1,
                                           "Risk": -1})
                    continue
                threshold = get_threshold(data)
                for sampler in pd.unique(data["sampler"]):
                    subset = data[data["sampler"] == sampler]
                    table_data.append({"Dataset": dataset, "OOD Detector": dsd, "Sample Size": sample_size,
                                       "Sampler": sampler,
                                       "FPR": fpr(subset, threshold=threshold),
                                       "DR": calibrated_detection_rate(subset, threshold=threshold),
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
    # print("vanilla")
    # collect_losswise_metrics("data/Polyp_ks_100_fullloss.csv")
    # print("knndsd")
    # collect_losswise_metrics("data/Polyp_ks_1NN_100_fullloss.csv")
    # experiment_prediction("data/imagenette_ks_5NN_500_fullloss.csv")
    # experiment_prediction("data/CIFAR10_ks_5NN_100_fullloss.csv")

    # experiment_prediction("data/NICO_ks_100_fullloss.csv")
    # experiment_prediction("data/CIFAR100_ks_5NN_100_fullloss.csv")
    # summarize_results()
    breakdown_by_sampler()
    # collect_losswise_metrics("Polyp_ks_10.csv")
    # collect_losswise_metrics("Polyp_ks_5NN_10.csv")
    # print("typicality")
    # risk("CIFAR_classifier_typicality_200_fullloss.csv")
    # print("vanilla")
    # risk("CIFAR_classifier_ks_200_fullloss.csv")
    # print("NN")
    # risk("CIFAR_classifier_ks_5NN_200_fullloss.csv")
    # print("")
    # print("5nn")
    # risk("NICO_classifier_ks_5NN_100_fullloss.csv")

    # for sample_size in [10, 20, 50, 100, 200, 500]:
    #     print(sample_size)
    #     risk(f"Njord_YOLO_ks_5NN_{sample_size}.csv")


    # print(calibrated_detection_rate(data[((data["fold"]!="dim")|(data["fold"]!="ind"))]["pvalue"], data[data["fold"]=="ind"]["pvalue"]))
    # print(auroc(data[((data["fold"]!="dim")|(data["fold"]!="ind"))]["pvalue"], data[data["fold"]=="ind"]["pvalue"]))
    # risk_across_noises("CIFAR10_ResNet_ks_200.csv")
    # for sample_size in [10, 20, 50, 100, 200, 500]:
    #     print(sample_size)
    #     risk(f"NICO_ResNet_ks_{sample_size}.csv")
    # print("mmd")
    # for sample_size in [10, 20, 50, 100, 200, 500]:
    #     print(sample_size)
    #     risk(f"NICO_ResNet_mmd_{sample_size}.csv")
