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
        torch.load("vae_logs/nico_dim/version_0/checkpoints/last.ckpt")[
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

def get_metrics():
    dataset = pd.read_csv("data.csv")
    for sample_size in np.unique(dataset["sample_size"]):
        subset = dataset[dataset["sample_size"] == sample_size]
        ood = subset[subset["ood_dataset"] != "nico_dim"]
        ind = subset[subset["ood_dataset"] == "nico_dim"]
        fpr_van = fprat95tpr(ood["vanilla_p"], ind["vanilla_p"])
        fpr_kn = fprat95tpr(ood["kn_p"], ind["kn_p"])
        # print("vanilla: ", fpr_van)
        # print("kn:", fpr_kn)
        aupr_van = aupr(ood["vanilla_p"], ind["vanilla_p"])
        aupr_kn = aupr(ood["kn_p"], ind["kn_p"])
        # print("vanilla: ", aupr_van)
        # print("kn:", aupr_kn)
        auroc_van = auroc(ood["vanilla_p"], ind["vanilla_p"])
        auroc_kn = auroc(ood["kn_p"], ind["kn_p"])
        # print("vanilla: ", auroc_van)
        # print("kn:", auroc_kn)
        corr_van = correlation(ood["vanilla_p"], ind["vanilla_p"], ood["loss"], ind["loss"])
        corr_kn = correlation(ood["kn_p"], ind["kn_p"], ood["loss"], ind["loss"])
        # print("vanilla: ", corr_van)
        # print("kn: ", corr_kn)
        dr_van = calibrated_detection_rate(ood["vanilla_p"], ind["vanilla_p"])
        dr_kn = calibrated_detection_rate(ood["kn_p"], ind["kn_p"])
        print("vanilla: ", dr_van)
        print("kn: ", dr_kn)


if __name__ == '__main__':
   # plot_nico_class_bias()
  get_metrics()