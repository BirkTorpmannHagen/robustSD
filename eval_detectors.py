import os

import matplotlib.pyplot as plt
import numpy as np

from vae.vae_experiment import VAEXperiment
from vae.models.vanilla_vae import VanillaVAE
import yaml
import torch
from torch.utils.data import DataLoader
from yellowbrick.features.manifold import Manifold
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle as pkl
from domain_datasets import build_nico_dataset
from classifier.resnetclassifier import ResNetClassifier
from scipy.stats import wasserstein_distance
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

            for i, (x, y, _) in tqdm(enumerate(DataLoader(ind_dataset)),total=len(ind_dataset)):
                with torch.no_grad():
                    latents[i] = self.rep_model.encode(x.to(self.config["device"])).cpu().numpy()
                    losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),y.to(self.config["device"])).cpu().numpy()
            pkl.dump(latents, open(fname_encodings, "wb"))
            pkl.dump(losses, open(fname_losses, "wb"))

        ood_latents = np.zeros((len(ood_dataset), self.rep_model.latent_dim))
        ood_losses = np.zeros(len(ood_dataset))
        for i, (x, y, _) in tqdm(enumerate(DataLoader(ood_dataset)), total=len(ood_dataset)):
            with torch.no_grad():
                ood_latents[i] = self.rep_model.encode(x.to(self.config["device"])).cpu().numpy()
                ood_losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),
                                                         y.to(self.config["device"])).cpu().numpy()

        # self.pca.fit(np.vstack((latents, ood_latents)))
        # latents = self.pca.transform(latents)
        # ood_latents = self.pca.transform(ood_latents)

        k_n_indx = [np.argmin(np.sum((np.expand_dims(i, 0) - latents) ** 2, axis=-1)) for i in ood_latents]
        k_nearest = latents[k_n_indx]
        # plt.scatter(latents[:, 0], latents[:, 1], label="ind")
        plt.scatter(ood_latents[:, 0], ood_latents[:, 1], label="ood")
        plt.scatter(k_nearest[:, 0], k_nearest[:, 1], label="kn")
        plt.legend()
        plt.show()

        p_vals_kn = []
        # p_vals_basic = []
        print(wasserstein_distance(ood_latents[:,0], k_nearest[:,0]))
        print(wasserstein_distance(ood_latents[:,1], k_nearest[:,1]))

        for j in range(25):
            sample_idx = np.random.choice(range(len(ood_latents)), sample_size)
            subsample_ind = k_nearest[sample_idx,:]
            subsample_ood = ood_latents[sample_idx,:]
            plt.scatter(subsample_ood[:,0],subsample_ood[:,1],label="ood")
            plt.scatter(subsample_ind[:, 0], subsample_ind[:, 1], label="ind")
            plt.legend()
            plt.title(str(j))
            plt.show()
            p_vals_kn.append(
                min(np.min([ks_2samp(subsample_ind[:,i], subsample_ood[:, i])[-1] for i in range(512)]) * 512, 1)
            )
            # p_vals_basic.append(
            #     min(np.min([ks_2samp(latents[:, i], ood_latents[sample_idx, i]) for i in range(2)]) * 2, 1)
            # )
        print(p_vals_kn)
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
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))

    ind = build_nico_dataset(1, "../../Datasets/NICO++", 0.1, trans, trans, context="dim", seed=0)[0]
    ood = build_nico_dataset(1, "../../Datasets/NICO++", 0.1, trans, trans, context="rock", seed=0)[1]
    ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.1, trans, trans, context="dim", seed=0)[1]

    config = yaml.safe_load(open("vae/configs/vae.yaml"))
    model = VanillaVAE(3, config["model_params"]["latent_dim"]).to("cuda")
    vae_exp = VAEXperiment(model, config)
    vae_exp.load_state_dict(
        torch.load("VAEs/nico_dim/version_0/checkpoints/last.ckpt")[
            "state_dict"])
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = ResNetClassifier.load_from_checkpoint("lightning_logs/version_2/checkpoints/epoch=55-step=559496.ckpt", num_classes=num_classes, resnet_version=34).to("cuda")
    aconfig = {"device":"cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    ds.compute_pvals_and_loss(ind, ood, 500)
    ds.compute_pvals_and_loss(ind, ind_val, 500)


