

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from bias_samplers import *

from tqdm import tqdm
import pickle as pkl
from domain_datasets import *

class BaseSD:
    def __init__(self):
        pass


class RabanserSD:
    def __init__(self) -> None:
        pass


class robustSD:
    def __init__(self, rep_model, classifier, config):
        self.rep_model = rep_model
        self.classifier = classifier
        self.config = config


    def compute_pvals_and_loss(self, ind_dataset, ood_dataset, sampler_name, sample_size, ind_dataset_name, ood_dataset_name, k=5, plot=False):
        sample_size = min(sample_size, len(ood_dataset))
        fname_encodings = f"robustSD_{ind_dataset_name}_enodings_{type(self.rep_model).__name__}.pkl"
        fname_losses = f"robustSD_{ind_dataset_name}_losses_{type(self.rep_model).__name__}.pkl"
        try:
            ind_latents = pkl.load(open(fname_encodings, "rb"))
            losses = pkl.load(open(fname_losses, "rb"))
        except FileNotFoundError:
            ind_latents = np.zeros((len(ind_dataset), self.rep_model.latent_dim))
            losses = np.zeros(len(ind_dataset))
            for i, (x, y, _) in tqdm(enumerate(DataLoader(ind_dataset)),total=len(ind_dataset)):
                with torch.no_grad():
                    ind_latents[i] = self.rep_model.encode(x.to(self.config["device"]))[0].cpu().numpy()
                    losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),y.to(self.config["device"])).cpu().numpy()
            pkl.dump(ind_latents, open(fname_encodings, "wb"))
            pkl.dump(losses, open(fname_losses, "wb"))

        ood_latents = np.zeros((len(ood_dataset), self.rep_model.latent_dim))
        ood_losses = np.zeros(len(ood_dataset))
        for i, (x, y, _) in tqdm(enumerate(ood_dataset), total=len(ood_dataset)):
            with torch.no_grad():
                ood_latents[i] = self.rep_model.encode(x.to(self.config["device"]))[0].cpu().numpy()

                ood_losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),
                                                         y.to(self.config["device"])).cpu().numpy()
        cols = ["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
        dataframe = []
        for start, stop in tqdm(list(zip(range(0, len(ood_dataset), sample_size), range(sample_size, len(ood_dataset)+sample_size, sample_size)))[:-1]): #perform tests
            sample_idx = range(start,stop)
            ood_samples = ood_latents[sample_idx]
            vanilla_pval = np.min([ks_2samp(ind_latents[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)])
            k_nearest_idx = np.concatenate([np.argpartition(np.sum((np.expand_dims(i, 0) - ind_latents) ** 2, axis=-1), k)[:k] for i in ood_samples])
            k_nearest_ind = ind_latents[k_nearest_idx]
            kn_pval = np.min([ks_2samp(k_nearest_ind[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)])
            dataframe.append(dict(zip(cols, [ind_dataset_name, ood_dataset_name, type(self.rep_model).__name__, sample_size, vanilla_pval, kn_pval, np.mean(ood_losses[sample_idx])])))
        final = pd.DataFrame(data=dataframe, columns=cols)
        return final

    def compute_pvals(self, ind_dataset, ood_dataset, ood_sampler, sample_size, ind_dataset_name, ood_dataset_name, k=5, plot=False):
        sample_size = min(sample_size, len(ood_dataset))
        fname_encodings = f"robustSD_{ind_dataset_name}_enodings_{type(self.rep_model).__name__}.pkl"
        try:
            ind_latents = pkl.load(open(fname_encodings, "rb"))
        except FileNotFoundError:
            ind_latents = np.zeros((len(ind_dataset), self.rep_model.latent_dim))
            for i, (x, y, _) in tqdm(enumerate(DataLoader(ind_dataset)),total=len(ind_dataset)):
                with torch.no_grad():
                    ind_latents[i] = self.rep_model.encode(x.to(self.config["device"]))[0].cpu().numpy()
            pkl.dump(ind_latents, open(fname_encodings, "wb"))

        ood_latents = np.zeros((len(ood_dataset), self.rep_model.latent_dim))
        for i, (x, y, _) in tqdm(enumerate(DataLoader(ood_dataset, sampler=ood_sampler)), total=len(ood_dataset)):
            with torch.no_grad():
                ood_latents[i] = self.rep_model.encode(x.to(self.config["device"]))[0].cpu().numpy()

        cols = ["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p"]
        dataframe = []
        for start, stop in list(zip(range(0, len(ood_dataset), sample_size), range(sample_size, len(ood_dataset)+sample_size, sample_size)))[:-1]: #perform tests
            sample_idx = range(start,stop)
            ood_samples = ood_latents[sample_idx]
            vanilla_pval = min(np.min([ks_2samp(ind_latents[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)]), 1)
            k_nearest_idx = np.concatenate([np.argpartition(np.sum((np.expand_dims(i, 0) - ind_latents) ** 2, axis=-1), k)[:k] for i in ood_samples])
            k_nearest_ind = ind_latents[k_nearest_idx]
            kn_pval = min(np.min([ks_2samp(k_nearest_ind[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)]), 1)
            dataframe.append(dict(zip(cols, [ind_dataset_name, ood_dataset_name, type(self.rep_model).__name__, sample_size, vanilla_pval, kn_pval])))
        final = pd.DataFrame(data=dataframe, columns=cols)
        return final

    def bootstrap_severity_estimation(self, ind_dataset, ood_dataset, ood_sampler, sample_size, ind_dataset_name, k=5, plot=False):
        #todo wip
        sample_size = min(sample_size, len(ood_dataset))
        fname_encodings = f"robustSD_{ind_dataset_name}_enodings_{type(self.rep_model).__name__}.pkl"
        fname_losses = f"robustSD_{ind_dataset_name}_losses_{type(self.rep_model).__name__}.pkl"
        try:
            ind_latents = pkl.load(open(fname_encodings, "rb"))
            losses = pkl.load(open(fname_losses, "rb"))
        except FileNotFoundError:
            ind_latents = np.zeros((len(ind_dataset), self.rep_model.latent_dim))
            losses = np.zeros(len(ind_dataset))
            for i, (x, y, _) in tqdm(enumerate(DataLoader(ind_dataset)),total=len(ind_dataset)):
                with torch.no_grad():
                    ind_latents[i] = self.rep_model.encode(x.to(self.config["device"]))[0].cpu().numpy()
                    losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),y.to(self.config["device"])).cpu().numpy()
            pkl.dump(ind_latents, open(fname_encodings, "wb"))
            pkl.dump(losses, open(fname_losses, "wb"))

        ood_latents = np.zeros((len(ood_dataset), self.rep_model.latent_dim))
        ood_losses = np.zeros(len(ood_dataset))
        for i, (x, y, _) in tqdm(enumerate(DataLoader(ood_dataset, sampler=ood_sampler)), total=len(ood_dataset)):
            with torch.no_grad():
                ood_latents[i] = self.rep_model.encode(x.to(self.config["device"]))[0].cpu().numpy()

                ood_losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),
                                                         y.to(self.config["device"])).cpu().numpy()
        cols = ["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
        dataframe = []
        for start, stop in list(zip(range(0, len(ood_dataset), sample_size), range(sample_size, len(ood_dataset)+sample_size, sample_size)))[:-1]: #perform tests
            sample_idx = range(start,stop)
            ood_samples = ood_latents[sample_idx]
            vanilla_pval = min(np.min([ks_2samp(ind_latents[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)]), 1)
            k_nearest_idx = np.concatenate([np.argpartition(np.sum((np.expand_dims(i, 0) - ind_latents) ** 2, axis=-1), k)[:k] for i in ood_samples])
            k_nearest_ind = ind_latents[k_nearest_idx]
            kn_pval = min(np.min([ks_2samp(k_nearest_ind[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)]), 1)
            dataframe.append(dict(zip(cols, [ind_dataset_name, "noise_level", type(self.rep_model).__name__, sample_size, vanilla_pval, kn_pval, np.mean(ood_losses[sample_idx])])))
        final = pd.DataFrame(data=dataframe, columns=cols)
        return final

    def eval_synthetic(self, ind_dataset, ind_val, trans_fn, sampler, sample_size, ind_dataset_name, ood_dataset_name, k=5, plot=False):
        dataset = transform_dataset(ind_val, trans_fn)
        return self.compute_pvals_and_loss(ind_dataset, dataset, ood_sampler=sampler,sample_size=sample_size, ind_dataset_name=ind_dataset_name, ood_dataset_name=ood_dataset_name, plot=plot, k=k)