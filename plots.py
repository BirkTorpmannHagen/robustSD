import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from vae.vae_experiment import VAEXperiment
from vae.models.vanilla_vae import VanillaVAE
import yaml
from metrics import *
from torch.utils.data import DataLoader
from domain_datasets import build_nico_dataset
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
from bias_samplers import *

def plot_nico_class_bias():
    config = yaml.safe_load(open("vae/configs/vae.yaml"))
    model = VanillaVAE(3, config["model_params"]["latent_dim"]).to("cuda")
    vae_exp = VAEXperiment(model, config)
    vae_exp.load_state_dict(
        torch.load("VAEs/nico_dim/version_0/checkpoints/last.ckpt")[
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

    pca.fit(np.vstack((ind_encodings, val_encodings)))
    transformed_ind = pca.transform(ind_encodings)
    transformed_val = pca.transform(val_encodings)

    for idx, (i,j) in enumerate(zip(np.linspace(0,int(.9*len(transformed_val)),10), np.linspace(int(0.1*len(transformed_val)), len(transformed_val), 10))):
        i = int(i)
        j = int(j)
        plt.scatter(transformed_ind[:,0], transformed_ind[:,1], label=f"InD")
        plt.scatter(transformed_val[i:j, 0],
                    transformed_val[i:j, 1], label=f"Biased")
        plt.legend()
        plt.savefig(f"figures/nico_clusterbias_{idx}.eps")
        plt.show()



if __name__ == '__main__':
    plot_nico_class_bias()
    # dataset = pd.read_csv("data.csv")
    # for sample_size in [10, 20, 50, 100, 200, 500, 1000, 10000]:
    #     subset = dataset[dataset["Sample Size"]==sample_size]
    #     kn = subset[subset["Method"]!="Vanilla"]
    #     vanilla = subset[subset["Method"]=="Vanilla"]
    #     kn_ind = kn[kn["Fold"]=="dim"]
    #     vanilla_ind = vanilla[vanilla["Fold"]=="dim"]
    #     kn_ood = kn[kn["Fold"]!="dim"]
    #     vanilla_ood = vanilla[vanilla["Fold"]!="dim"]
    #     sns.boxplot(data=kn,x="Fold", y="P")
    #     plt.show()
    #     # vanilla_ood.loc[:, "P"]*=256
    #     # vanilla_ind.loc[:,"P"]*=256
    #
    #     # input()
    #     # print(sample_size)
    #     fpr_van = fprat95tpr(vanilla_ood, vanilla_ind)
    #     fpr_kn = fprat95tpr(kn_ood, kn_ind)
    #     print("fpr95tpr vanilla: ", fpr_van)
    #     print("fpr95tpr kn: ", fpr_kn)
    #     # # plt.hist(kn_ood["P"], label="ood")
    #     # # plt.hist(kn_ind["P"], label="ind")
    #     # # plt.title("kn")
    #     # # plt.legend()
    #     # # plt.show()
    #     # # plt.hist(vanilla_ood["P"], label="ood")
    #     # # plt.hist(vanilla_ind["P"], label="ind")
    #     # # plt.title("vanilla")
    #     # # plt.legend()
    #     # # plt.show()
    #     # auc_van = auroc(vanilla_ood, vanilla_ind)
    #     # print("vanilla auc: ", auc_van)
    #     # auc_kn = auroc(kn_ood, kn_ind)
    #     # print("kn auc: ", auc_kn)
    #     # auc_van = aupr(vanilla_ood, vanilla_ind)
    #     # print("vanilla aupr: ", auc_van)
    #     # auc_kn = aupr(kn_ood, kn_ind)
    #     # print("kn aupr: ", auc_kn)