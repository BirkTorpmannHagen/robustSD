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
from tqdm import tqdm
import pickle as pkl


def generate_plot(dataloaders, fold_names):
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
    datasets = create_datasets_by_fold()
    kn_acc=[]
    basic_acc=[]
    kn_dr = []
    v_dr = []
    sample_range = range(100, 2500, 100)
    for sample_size in sample_range:
        kn_tpn = 0
        basic_tpn = 0
        total = 3000
        kn_fold_prec = [0, 0, 0]
        basic_fold_prec = [0, 0, 0]
        for i, fold in enumerate(["ind_val", "ood", "test_val"]):
            kn_pval, basic_pval = sample_p_vals(datasets, ["ind", fold], 1000, sample_size)
            if fold == "ind_val":
                kn_tpn += (kn_pval > 0.05).astype(int).sum()
                kn_fold_prec[i] = (kn_pval > 0.05).astype(int).mean()
                basic_fold_prec[i] = (basic_pval > 0.05).astype(int).mean()
                basic_tpn += (basic_pval > 0.05).astype(int).sum()
            else:
                kn_tpn += (kn_pval < 0.05).astype(int).sum()
                kn_fold_prec[i] = (kn_pval < 0.05).astype(int).mean()
                basic_fold_prec[i] = (basic_pval < 0.05).astype(int).mean()
                basic_tpn += (basic_pval < 0.05).astype(int).sum()
        kn_dr.append(kn_fold_prec)
        v_dr.append(basic_fold_prec)
        kn_acc.append(kn_tpn / total)
        basic_acc.append(basic_tpn / total)
    kn_dr = np.array(kn_dr)
    v_dr = np.array(v_dr)
    plt.rcParams["figure.figsize"] = (10, 4)

    for i, fold in enumerate(["ind_val", "ood", "test_val"]):
        plt.plot(sample_range, kn_dr[:, i], label=fold)
    plt.plot(sample_range, kn_acc, label="Total Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("acc")
    print(kn_acc)
    print(basic_acc)
