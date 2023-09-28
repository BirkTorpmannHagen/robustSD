import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from vae.vae_experiment import VAEXperiment
from vae.models.vanilla_vae import VanillaVAE, ResNetVAE
from classifier.resnetclassifier import ResNetClassifier
import yaml
from metrics import *
from bias_samplers import ClusterSampler, ClusterSamplerWithSeverity, ClassOrderSampler
from torch.utils.data import DataLoader
from domain_datasets import build_nico_dataset
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import os
from bias_samplers import *


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
            fpr_van = fprat95tpr(ood["vanilla_p"], ind["vanilla_p"])
            fpr_kn = fprat95tpr(ood["kn_p"], ind["kn_p"])
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
            fpr_van = fprat95tpr(ood["vanilla_p"], ind["vanilla_p"])
            fpr_kn = fprat95tpr(ood["kn_p"], ind["kn_p"])
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
                fpr_van = fprat95tpr(ood["vanilla_p"], ind["vanilla_p"])
                fpr_kn = fprat95tpr(ood["kn_p"], ind["kn_p"])
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
            fpr_van = fprat95tpr(ood["vanilla_p"], ind["vanilla_p"])
            fpr_kn = fprat95tpr(ood["kn_p"], ind["kn_p"])
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

if __name__ == '__main__':

    risk("CIFAR_classifier_ks_100_fullloss.csv")
    risk("CIFAR_classifier_ks_5NN_100_fullloss.csv")
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
