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
from domain_datasets import build_nico_dataset, wrap_dataset
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

def get_classification_metrics(filename):
    dataset = pd.read_csv(filename)
    for sample_size in np.unique(dataset["sample_size"]):
        subset = dataset[dataset["sample_size"] == sample_size]
        ood = subset[subset["ood_dataset"] != subset["ind_dataset"]]
        ind = subset[subset["ood_dataset"] == subset["ind_dataset"]]
        # plt.hist(ind["vanilla_p"], label="vanilla")
        # plt.hist(ind["kn_p"], label="kn")
        # print(subset.groupby(["ood_dataset"])["vanilla_p"].mean())
        # print(subset.groupby(["ood_dataset"])["kn_p"].mean())
        # input()
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
        # corr_van = correlation(ood["vanilla_p"], ind["vanilla_p"], ood["loss"], ind["loss"])
        # corr_kn = correlation(ood["kn_p"], ind["kn_p"], ood["loss"], ind["loss"])
        # print("vanilla: ", corr_van)
        # print("kn: ", corr_kn)
        # f, ax = plt.subplots(figsize=(7, 7))
        # ax.set(yscale="log")
        # sns.regplot(np.concatenate((ood["loss"], ind["loss"])), np.concatenate((ood["kn_p"], ind["kn_p"])), ax=ax, color="blue")
        # ax2 = plt.twinx()
        # ax.set_ylabel("kn_p", color="blue", fontsize=14)
        # ax.set_ylabel("vanilla_p", color="orange", fontsize=14)
        # sns.regplot(np.concatenate((ood["loss"], ind["loss"])), np.concatenate((ood["vanilla_p"], ind["vanilla_p"])), ax=ax2, color="orange")
        # plt.legend()
        # plt.title(f"{corr_kn} at n={sample_size}")
        # plt.show()
        dr_van = calibrated_detection_rate(ood["vanilla_p"], ind["vanilla_p"])
        dr_kn = calibrated_detection_rate(ood["kn_p"], ind["kn_p"])
        print("vanilla DR: ", dr_van)
        print("kn DR: ", dr_kn)


def get_corrrelation_metrics(filename_noise, filename_ood):
    dataset = pd.read_csv(filename_noise)
    # raw correlation data
    for sample_size in np.unique(dataset["sample_size"]):
        subset = dataset[dataset["sample_size"] == sample_size]
        corr_van = correlation(np.log10(subset["vanilla_p"]), subset["loss"])
        corr_kn = correlation(np.log10(subset["kn_p"]), subset["loss"])
        print("correlation vanilla", corr_van)
        print("correlation kn", corr_kn)
        f, ax = plt.subplots(figsize=(7, 7))
        sns.regplot(subset["kn_p"],subset["loss"], ax=ax, color="blue")
        ax.set_xlabel("kn_p", color="blue", fontsize=14)
        ax.set(xscale="log")

        # sns.regplot(subset["vanilla_p"], subset["loss"],  ax=ax2, color="orange")
        plt.ylim((0,10))
        plt.legend()
        plt.title(f"{round(corr_kn[0],3)} at n={sample_size}")
        plt.show()

        f2, ax2 = plt.subplots(figsize=(7, 7))
        ax2.set(xscale="log")
        sns.regplot(subset["vanilla_p"], subset["loss"],  ax=ax2, color="orange")
        plt.ylim((0,10))
        plt.show()
    # predictive
    dataset_ood = pd.read_csv(filename_ood)
    for sample_size in np.unique(dataset["sample_size"]):
        subset = dataset[dataset["sample_size"] == sample_size]
        vanilla_predictive_likelihood = get_loss_pdf_from_ps(subset["vanilla_p"], subset["loss"], dataset_ood["vanilla_p"], dataset_ood["loss"])
        kn_predictive_likelihood = get_loss_pdf_from_ps(subset["kn_p"], subset["loss"], dataset_ood["kn_p"], dataset_ood["loss"])
        print("vanilla predictive likelihood", vanilla_predictive_likelihood)
        print("kn predictive likelihood", kn_predictive_likelihood)

def genfailure_metrics(filename):
    dataset = pd.read_csv(filename)
    for sample_size in np.unique(dataset["sample_size"]):
        subset = dataset[dataset["sample_size"] == sample_size]
        ood = subset[subset["ood_dataset"] != "nico_dim"]
        ind = subset[subset["ood_dataset"] == "nico_dim"]
        print(sample_size)
        print("ood: ", ood["loss"].mean())
        print("ind: ", ind["loss"].mean())


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
    das_kn = []
    das_vn = []
    dataset = pd.read_csv(filename)
    for sample_size in np.unique(dataset["sample_size"]):
        subset = dataset[dataset["sample_size"] == sample_size]
        ood = subset[subset["ood_dataset"] != "nico_dim"]
        ind = subset[subset["ood_dataset"] == "nico_dim"]
        das_kn.append(calibrated_detection_rate(ood["kn_p"], ind["kn_p"]))
        das_vn.append(calibrated_detection_rate(ood["vanilla_p"], ind["vanilla_p"]))
    plt.plot(np.unique(dataset["sample_size"]), das_kn, label="kn")
    plt.plot(np.unique(dataset["sample_size"]), das_vn, label="vanilla")
    plt.legend()
    plt.show()

def eval_k_impact(filename):
    #todo collect a csv with k-data for some dataset and sample size
    das_kn = []
    das_vn = []
    dataset = pd.read_csv(filename)
    for k in np.unique(dataset["k"]):
        subset = dataset[dataset["k"] == k]
        ood = subset[subset["ood_dataset"] != "nico_dim"]
        ind = subset[subset["ood_dataset"] == "nico_dim"]
        das_kn.append(calibrated_detection_rate(ood["kn_p"], ind["kn_p"]))
        das_vn.append(calibrated_detection_rate(ood["vanilla_p"], ind["vanilla_p"]))
    plt.plot(np.unique(dataset["k"]), das_kn, label="kn")
    plt.plot(np.unique(dataset["k"]), das_vn, label="vanilla")
    plt.legend()
    plt.show()

if __name__ == '__main__':
  # plot_loss_v_encodings()
  # plot_nico_clustering_bias()
  # plot_nico_class_bias()
  # genfailure_metrics("ResNetClassifier_dim_k5_ClassOrderSampler.csv") #potential bu88
  get_classification_metrics("CVC_ClinicDB_ResNetVAE_k5_ClusterSampler.csv") #lower k is slightly better with class-bias?
  # genfailure_metrics()
  # get_classification_metrics("ResNetClassifier_dim_k10_ClassOrderSampler.csv")
  # get_corrrelation_metrics("lp_data_nico_noise.csv")
  # from torchvision.datasets import MNIST
  # from torchvision.datasets import CIFAR10,CIFAR100
  # dataset = CIFAR10("~/Datasets/cifar10", train=True, download=True)
  # for x,y, z in dataset:
  #   plt.imshow(x)
  #   plt.show()
  #   break
  # CIFAR100("~/Datasets/cifar100", train=True, download=True)
  # MNIST("~/Datasets/mnist", train=True, download=True)
  # num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
  #
  # model = ResNetClassifier.load_from_checkpoint(
  #     "/home/birk/Projects/robustSD/lightning_logs/version_0/checkpoints/epoch=109-step=1236510.ckpt",
  #     num_classes=num_classes, resnet_version=34).to("cuda")
  #
  # trans = transforms.Compose([transforms.RandomHorizontalFlip(),
  #                             transforms.Resize((512, 512)),
  #                             transforms.ToTensor(), ])
  # ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
  # for i, (x,y,_) in enumerate(DataLoader(ind_val, sampler=ClusterSamplerWithSeverity(ind_val,model, 50, 0.5 ))):
  #     pass
  #