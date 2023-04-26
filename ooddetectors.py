

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from bias_samplers import *

from tqdm import tqdm
import pickle as pkl
from domain_datasets import *
from vae.models.vanilla_vae import VanillaVAE

class BaseSD:
    def __init__(self, rep_model, sample_selector):
        self.sample_selector = sample_selector
        self.rep_model = rep_model

    def register_testbed(self, testbed):
        self.testbed = testbed


class RabanserSD:
    def __init__(self, sample_selector) -> None:
        pass

class TypicalitySD(BaseSD):
    def __init__(self,rep_model, sample_selector):
        super().__init__(rep_model, sample_selector)

    def compute_entropy(self, samples):
        pass

    def compute_pvals_and_loss(self, sample_size, mc_samples=25):
        sample_size=(sample_size, len(self.testbed.ood_loader()))
        # resubstitution estimation of entropy
        try:
            likelihoods = torch.load(f"{type(self).__name__}_{type(self.testbed).__name__}.pt")
        except FileNotFoundError:
            likelihoods = []
            for x,y,_ in self.testbed.ind_loader():
                likelihoods.append(self.rep_model.monte_carlo_likelihood_estimate(x, mc_samples))
            torch.save(likelihoods, f"{type(self).__name__}_{type(self.testbed).__name__}.pt")
        resub_entropy = -torch.log(torch.tensor(likelihoods)).mean().item()

        # compute ind_val pvalues
        ind_likelihoods = []
        for x,y,_ in self.testbed.ind_val_loader():
            ind_likelihoods.append(self.rep_model.monte_carlo_likelihood_estimate(x, mc_samples))
        ind_val_entropy = -torch.log(torch.tensor(ind_likelihoods))

        ood_likelihoods = [[[],[],[]] for i in self.testbed.ood_loaders()]
        for i, ood_set in enumerate(self.testbed.ood_loaders()):
            for j, ood_wsampler in ood_set:
                for x,y,_ in ood_wsampler: #todo possible bug here
                    ood_likelihoods[i][j].append(self.rep_model.monte_carlo_likelihood_estimate(x, mc_samples))
        ood_entropies = [-torch.log(torch.tensor(ood_set)) for ood_set in ood_likelihoods]

        print("Resubstitution entropy: ", resub_entropy)
        print("Ind val entropy: ", ind_val_entropy.mean().item())
        ood_pvals = [[[],[],[]] for i in self.testbed.ood_loaders()]


        #bootstrap from ind_val to generate a distribution of entropies used to compute pvalues
        bootstrap_entropy_distribution = [np.random.choice(ind_val_entropy, sample_size).mean().item() for i in range(1000)]
        entropy_epsilon = np.quantile(bootstrap_entropy_distribution, 0.99) # alpha of .99 quantile
        print("Entropy epsilon: ", entropy_epsilon)
        for ood_entropy_noiseval in ood_entropies:
            for ood_entropy_sampler in ood_entropy_noiseval:
                for x,y, _ in ood_entropy_sampler:
                    #batch size determined by sample_size
                    #get p_vals for each batch by comparing to bootstrap distribution

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