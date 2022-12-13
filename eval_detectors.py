import os

import matplotlib.pyplot as plt
import numpy as np

from vae.vae_experiment import VAEXperiment
from vae.models.vanilla_vae import VanillaVAE
import yaml
import torch
from yellowbrick.features.manifold import Manifold
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle as pkl
from domain_datasets import build_dataset

class robustSD:
    def __init__(self, rep_model, classifier, config):
        self.rep_model = rep_model
        self.classifier = classifier
        self.config = config
        self.pca = PCA(2)


    def compute_pvals_and_loss(self, ind_dataset, ood_dataset, sample_size):
        sample_size = min(sample_size, len(ood_dataset))
        ind_dataset_name = ""
        ood_dataset_name = ""
        fname_encodings = f"robustSD_{ind_dataset_name}_enodings"
        fname_losses = f"robustSD_{ind_dataset_name}_losses"

        try:
            latents = pkl.load(open(fname_encodings, "rb"))
            losses = pkl.load(open(fname_losses, "rb"))
        except FileNotFoundError:
            latents = np.zeros((len(ind_dataset), self.rep_model.latent_dim))
            losses = np.zeros(len(ind_dataset))

            for i, (x, y, _) in ind_dataset:
                latents[i] = self.rep_model.encode(x.to(self.config["device"])).cpu().numpy()
                losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),y.to(self.config["device"])).cpu().numpy()
            pkl.dump(latents, open(fname_encodings, "wb"))
            pkl.dump(losses, open(fname_losses, "wb"))

        ood_latents = np.zeros((len(ood_dataset), self.rep_model.latent_dim))
        ood_losses = np.zeros(len(ood_dataset))
        for i, (x, y, _) in ood_dataset:
            ood_latents[i] = self.rep_model.encode(x.to(self.config["device"])).cpu().numpy()
            ood_losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),
                                                     y.to(self.config["device"])).cpu().numpy()

        self.pca.fit(np.vstack(latents, ood_latents))
        latents = self.pca.transform(latents)
        ood_latents = self.pca.transform(ood_latents)

        k_n_indx = [np.argmin(np.sum((np.expand_dims(i, 0) - latents) ** 2, axis=-1)) for i in ood_latents]
        k_nearest = latents[k_n_indx]

        p_vals_kn = []
        # p_vals_basic = []
        for i in range(1000):
            sample_idx = np.random.choice(range(len(ood_latents)), sample_size)
            p_vals_kn.append(
                min(np.min([ks_2samp(k_nearest[sample_idx, i], ood_latents[sample_idx, i]) for i in range(2)]) * 2, 1)
            )
            # p_vals_basic.append(
            #     min(np.min([ks_2samp(latents[:, i], ood_latents[sample_idx, i]) for i in range(2)]) * 2, 1)
            # )
        return np.array(p_vals_kn), (losses, ood_losses)


    def bootstrap_severity_estimation(self):
        pass

    def eval_given_transforms(self, dataset, transforms):
        for transform in transforms:
            pass


def generate_plot(dataloaders, fold_names):
    assert len(dataloaders)==len(fold_names), "not enough names"
    model = VanillaVAE(3, 128).to("cuda")
    config = yaml.safe_load("configs/vae.yaml")
    vae_exp = VAEXperiment(model, config)
    vae_exp.load_state_dict(
        torch.load("lightning_logs/version_1/checkpoints/epoch=42-step=2752215.ckpt")[
            "state_dict"])
    try:
        ind_latents = pkl.load(open("latents_vis_ind.pkl", "rb"))
        ood_latents = pkl.load(open(f"latents_vis_{fold_names[-1]}.pkl", "rb"))
    except FileNotFoundError:
        ind_latents =  np.zeros((len(dataloaders[0]), 128))
        ood_latents = np.zeros((len(dataloaders[1]), 128))
        for didx, dlaoder in enumerate(dataloaders):
            try:
                ind_latents = pkl.load(open("latents_vis_ind.pkl", "rb"))
                if didx==0:
                    continue
            except FileNotFoundError:
                pass
            with torch.no_grad():
                for i, (img, y, paths, _) in tqdm(enumerate(dlaoder),total=len(dlaoder)):
                    # TODO implement checks to make sure this is not the val set
                    x = img.to("cuda").float() / 255
                    if didx==0:
                        ind_latents[i] = model.encode(x)[0].cpu().numpy()
                    elif didx==1:
                        ood_latents[i] = model.encode(x)[0].cpu().numpy()

    pkl.dump(ind_latents, open(f"latents_vis_ind.pkl", "wb"))
    pkl.dump(ood_latents, open(f"latents_vis_{fold_names[-1]}.pkl", "wb"))
    ind = ind_latents
    ood = ood_latents

    tsne = PCA(2)
    samples = tsne.fit_transform(np.vstack((ind, ood)))
    ind =samples[:len(dataloaders[0]), :]
    ood = samples[len(dataloaders[0]):, :]

    # k_n_indx = np.argsort(np.sum((np.expand_dims(np.mean(ood, axis=0),0)-ind)**2, axis=-1))[:32]
    k_n_indx = [np.argmin(np.sum((np.expand_dims(i,0)-ind)**2, axis=-1)) for i in ood]
    k_nearest = ind[k_n_indx]
    plt.scatter(ind[:,0], ind[:,1], label="ind",s=2, c="gray")
    plt.scatter(ood[:,0], ood[:,1], label="test_val")
    # plt.scatter(k_nearest[:, 0], k_nearest[:, 1], label="k_nearest")
    plt.legend()
    plt.savefig("test_val.eps")
    print(np.min([ks_2samp(k_nearest[:, i], ood[:, i]) for i in range(2)])*128)
    print(np.min([ks_2samp(ind[:, i], ood[:, i]) for i in range(2)])*128)

    plt.show()
    print(samples.shape)
    # tsne.show()

def sample_p_vals(dataloaders, fold_names, n_samples, test_samples):
    assert len(dataloaders)==len(fold_names), "not enough names"
    model = VanillaVAE(3, 128).to("cuda")
    config = yaml.safe_load("configs/vae.yaml")
    vae_exp = VAEXperiment(model, config)
    vae_exp.load_state_dict(
        torch.load("/home/birk/BatchDiversitySampling/logs/VanillaVAE/version_0/checkpoints/epoch=16-step=31109.ckpt")[
            "state_dict"])
    try:
        ind_latents = pkl.load(open("latents_vis_ind.pkl", "rb"))
        ood_latents = pkl.load(open(f"latents_vis_{fold_names[-1]}.pkl", "rb"))
    except FileNotFoundError:
        ind_latents =  np.zeros((len(dataloaders[0]), 128))
        ood_latents = np.zeros((len(dataloaders[1]), 128))
        for didx, dlaoder in enumerate(dataloaders):
            try:
                ind_latents = pkl.load(open("latents_vis_ind.pkl", "rb"))
                if didx==0:
                    continue
            except FileNotFoundError:
                pass
            with torch.no_grad():
                for i, (img, y, paths, _) in tqdm(enumerate(dlaoder),total=len(dlaoder)):
                    # TODO implement checks to make sure this is not the val set
                    x = img.to("cuda").float() / 255
                    if didx==0:
                        ind_latents[i] = model.encode(x)[0].cpu().numpy()
                    elif didx==1:
                        ood_latents[i] = model.encode(x)[0].cpu().numpy()

    pkl.dump(ind_latents, open(f"latents_vis_ind.pkl", "wb"))
    pkl.dump(ood_latents, open(f"latents_vis_{fold_names[-1]}.pkl", "wb"))
    ind = ind_latents
    ood = ood_latents

    tsne = PCA(2)
    samples = tsne.fit_transform(np.vstack((ind, ood)))
    # samples = np.vstack((ind, ood))
    ind =samples[:len(dataloaders[0]), :]
    ood = samples[len(dataloaders[0]):, :]
    print(ind.shape)
    print(ood.shape)
    # k_n_indx = np.argsort(np.sum((np.expand_dims(np.mean(ood, axis=0),0)-ind)**2, axis=-1))[:32]
    k_n_indx = [np.argmin(np.sum((np.expand_dims(i,0)-ind)**2, axis=-1)) for i in ood]
    k_nearest = ind[k_n_indx]

    p_vals_kn = []
    p_vals_basic = []
    for i in range(n_samples):
        sample_idx = np.random.choice(range(len(ood)), test_samples)
        p_vals_kn.append(
            min(np.min([ks_2samp(k_nearest[sample_idx, i], ood[sample_idx, i]) for i in range(2)]) * 2, 1)
        )
        p_vals_basic.append(
            min(np.min([ks_2samp(ind[:, i], ood[sample_idx, i]) for i in range(2)]) * 2, 1)
        )
    return np.array(p_vals_kn), np.array(p_vals_basic)





if __name__ == '__main__':
    # generate_plot(create_datasets_by_fold(), ["ind", "test_val"])
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
    datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))



    sample_range = range(100, 2500, 100)
    for sample_size in sample_range:
        for i, fold in enumerate(["ind_val", "ood", "test_val"]):
            kn_pval, basic_pval = sample_p_vals(datasets, ["ind", fold], 1000, sample_size)


