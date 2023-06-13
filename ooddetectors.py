import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn

from scipy.stats import ks_2samp

import metrics
from bias_samplers import *

from tqdm import tqdm
import pickle as pkl
from domain_datasets import *
from vae.models.vanilla_vae import VanillaVAE
import torch_two_sample as tts

class BaseSD:
    def __init__(self, rep_model, sample_selector):
        self.sample_selector = sample_selector
        self.rep_model = rep_model

    def register_testbed(self, testbed):
        self.testbed = testbed


class RabanserSD(BaseSD):
    def __init__(self, rep_model, sample_selector):
        super().__init__(rep_model, sample_selector)

    def get_encodings(self, dataloader):
        encodings = np.zeros((len(dataloader), self.rep_model.latent_dim))
        for i, (x, y, _) in enumerate(dataloader):
            with torch.no_grad():
                x = x.to("cuda")
                encodings[i] = self.rep_model.get_encoding(x).cpu().numpy() #mu from vae or features from classifier
        return encodings

    def compute_pvals_and_loss_for_loader(self,ind_encodings, dataloaders, sample_size):
        encodings = dict([(loader.sampler.__class__.__name__, self.get_encodings(loader)) for loader in dataloaders])
        losses = dict([(loader.sampler.__class__.__name__, self.testbed.compute_losses(loader)) for loader in dataloaders])

        p_values = dict([(biased_sampler_name, []) for biased_sampler_name in encodings.keys()])
        sample_losses = dict([(biased_sampler_name, []) for biased_sampler_name in encodings.keys()])

        for biased_sampler_name, biased_sampler_encodings in encodings.items():
            for start, stop in list(zip(range(0, len(biased_sampler_encodings), sample_size),
                                        range(sample_size, len(biased_sampler_encodings) + sample_size, sample_size)))[
                               :-1]:
                ood_samples = biased_sampler_encodings[start:stop]
                p_value = np.min([ks_2samp(ind_encodings[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)])
                p_values[biased_sampler_name].append(p_value)
                sample_losses[biased_sampler_name].append(np.mean(
                    losses[biased_sampler_name][start:stop]))

        return p_values, sample_losses


    def compute_pvals_and_loss(self, sample_size):
        """

        :param sample_size: sample size for the tests
        :return: ind_p_values: p-values for ind fold for each sampler
        :return ood_p_values: p-values for ood fold for each sampler
        :return ind_sample_losses: losses for each sampler on ind fold, in correct order
        :return ood_sample_losses: losses for each sampler on ood fold, in correct order
        """
        # sample_size = min(sample_size, len(self.testbed.ind_val_loaders()[0]))
        try:
            ind_latents = torch.load(f"{type(self).__name__}_{type(self.testbed).__name__}.pt")
        except FileNotFoundError:
            ind_latents = self.get_encodings(self.testbed.ind_loader())
            torch.save(ind_latents, f"{type(self).__name__}_{type(self.testbed).__name__}.pt")

        ind_pvalues, ind_losses = self.compute_pvals_and_loss_for_loader(ind_latents, self.testbed.ind_val_loaders(), sample_size)
        ood_pvalues_dict = {}
        ood_losses_dict = {}
        for fold, ood_loader in self.testbed.ood_loaders().items():
            ood_pvalues, ood_losses = self.compute_pvals_and_loss_for_loader(ind_latents, [ood_loader], sample_size)
            ood_pvalues_dict[fold] = (ood_pvalues, ood_losses)

        return ind_pvalues, ood_pvalues_dict, ind_losses, ood_losses_dict


class TypicalitySD(BaseSD):
    def __init__(self,rep_model, sample_selector):
        super().__init__(rep_model, sample_selector)

    def compute_entropy(self, data_loader):
        log_likelihoods = []
        for i, (x, y, _) in enumerate(data_loader):
            x = x.to("cuda")
            log_likelihoods.append(self.rep_model.estimate_log_likelihood(x))
        entropies = -(torch.tensor(log_likelihoods))
        return entropies

    # def compute_pvals_and_loss_for_loader(self, ind_entropy, dataloaders, sample_size):


    def compute_pvals_and_loss(self, sample_size):
        """

        :param sample_size: sample size for the tests
        :return: ind_p_values: p-values for ind fold for each sampler
        :return ood_p_values: p-values for ood fold for each sampler
        :return ind_sample_losses: losses for each sampler on ind fold, in correct order
        :return ood_sample_losses: losses for each sampler on ood fold, in correct order
        """
        # sample_size=min(sample_size, len(self.testbed.ind_val_loaders()[0]))

        # resubstitution estimation of entropy
        try:
            loglikelihoods = torch.load(f"{type(self).__name__}_{type(self.testbed).__name__}.pt")
        except FileNotFoundError:
            loglikelihoods = []
            for x,y,_ in self.testbed.ind_loader():
                x = x.to("cuda")
                loglikelihoods.append(self.rep_model.estimate_log_likelihood(x))
            torch.save(loglikelihoods, f"{type(self).__name__}_{type(self.testbed).__name__}.pt")
        resub_entropy = -(torch.tensor(loglikelihoods)).mean().item()

        # compute ind_val pvalues for each sampler
        ind_val_entropies = dict([(loader.sampler.__class__.__name__, self.compute_entropy(loader)) for loader in self.testbed.ind_val_loaders()])
        ind_val_losses = dict([(loader.sampler.__class__.__name__, self.testbed.compute_losses(loader)) for loader in self.testbed.ind_val_loaders()])
        print(ind_val_entropies)
        #bootstrap from ind_val to generate a distribution of entropies used to compute pvalues
        ind_val_entropy_nobias = ind_val_entropies["RandomSampler"] #assume bootstrapping from unbiased data
        bootstrap_entropy_distribution = sorted([np.random.choice(ind_val_entropy_nobias, sample_size).mean().item() for i in range(10000)])
        entropy_epsilon = np.quantile(bootstrap_entropy_distribution, 0.99) # alpha of .99 quantile

        ind_p_values = dict([(biased_sampler_name, []) for biased_sampler_name in ind_val_entropies.keys()])
        ind_sample_losses = dict([(biased_sampler_name, []) for biased_sampler_name in ind_val_entropies.keys()])

        for biased_sampler_name, biased_sampler_entropies in ind_val_entropies.items():
            for start, stop in list(zip(range(0, len(biased_sampler_entropies), sample_size),
                                range(sample_size, len(biased_sampler_entropies)+sample_size, sample_size)))[:-1]:
                sample_entropy = torch.mean(biased_sampler_entropies[start:stop])
                p_value = 1-np.mean([1 if sample_entropy > i else 0 for i in bootstrap_entropy_distribution])
                ind_p_values[biased_sampler_name].append(p_value)
                ind_sample_losses[biased_sampler_name].append(np.mean(ind_val_losses[biased_sampler_name][start:stop]))


        # compute ood pvalues
        ood_pvalues = dict([(loader.sampler.__class__.__name__, []) for loader in self.testbed.ood_loaders()])
        ood_sample_losses = dict([(loader.sampler.__class__.__name__, []) for loader in self.testbed.ood_loaders()])
        import matplotlib.pyplot as plt
        for i, ood_set in enumerate(self.testbed.ood_loaders()):
                entropies=self.compute_entropy(ood_set)
                losses = self.testbed.compute_losses(ood_set)
                for start, stop in list(zip(range(0, len(entropies), sample_size),
                                            range(sample_size, len(entropies) + sample_size, sample_size)))[:-1]:
                    sample_entropy = torch.mean(entropies[start:stop])
                    p_value = 1-np.mean([1 if sample_entropy > entropy_test else 0 for entropy_test in bootstrap_entropy_distribution])
                    ood_pvalues[ood_set.sampler.__class__.__name__].append(p_value)
                    ood_sample_losses[ood_set.sampler.__class__.__name__].append(np.mean(losses[start:stop]))

        # for sampler in ood_pvalues.keys():
        #     ood_pvalues_by_sampler = np.array(ood_pvalues[sampler])
        #     ind_pvalues_by_sampler = np.array(ind_p_values[sampler])
        return ind_p_values, ood_pvalues, ind_sample_losses, ood_sample_losses



class robustSD:
    def __init__(self, rep_model, classifier, config):
        self.rep_model = rep_model
        self.classifier = classifier
        self.config = config


    def compute_pvals_and_loss(self, ind_dataset, ood_dataset, sample_size, ind_dataset_name, ood_dataset_name, k=5, plot=False):
        sample_size = min(sample_size, len(ood_dataset))
        fname_encodings = f"robustSD_{ind_dataset_name}_enodings_{type(self.rep_model).__name__}.pkl"
        fname_losses = f"robustSD_{ind_dataset_name}_losses_{type(self.rep_model).__name__}.pkl"
        try:
            ind_latents = torch.load(fname_encodings)
            losses = torch.load(fname_losses)
        except FileNotFoundError:
            ind_latents = np.zeros((len(ind_dataset), self.rep_model.latent_dim))
            losses = np.zeros(len(ind_dataset))
            for i, (x, y, _) in tqdm(enumerate(ind_dataset),total=len(ind_dataset)):
                with torch.no_grad():
                    ind_latents[i] = self.rep_model.get_encoding(x.to(self.config["device"])).cpu().numpy()
                    losses[i] = torch.nn.MSELoss()(self.classifier(x.to(self.config["device"])),y.to(self.config["device"])).item()
            torch.save(ind_latents, fname_encodings)
            torch.save(losses, fname_losses)

        ood_latents = np.zeros((len(ood_dataset), self.rep_model.latent_dim))
        ood_losses = np.zeros(len(ood_dataset))
        for i, (x, y, _) in tqdm(enumerate(ood_dataset), total=len(ood_dataset)):
            with torch.no_grad():
                ood_latents[i] = self.rep_model.get_encoding(x.to(self.config["device"])).cpu().numpy()

                ood_losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),
                                                         y.to(self.config["device"])).cpu().numpy()
        cols = ["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
        dataframe = []
        for start, stop in tqdm(list(zip(range(0, len(ood_dataset), sample_size), range(sample_size, len(ood_dataset)+sample_size, sample_size)))[:-1]): #perform tests
            sample_idx = range(start,stop)
            ood_samples = ood_latents[sample_idx]
            vanilla_pval = np.min([ks_2samp(ind_latents[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)])
            print(vanilla_pval)
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
                    ind_latents[i] = self.rep_model.get_encoding(x.to(self.config["device"])).cpu().numpy()
            pkl.dump(ind_latents, open(fname_encodings, "wb"))

        ood_latents = np.zeros((len(ood_dataset), self.rep_model.latent_dim))
        for i, (x, y, _) in tqdm(enumerate(DataLoader(ood_dataset, sampler=ood_sampler)), total=len(ood_dataset)):
            with torch.no_grad():
                ood_latents[i] = self.rep_model.get_encoding(x.to(self.config["device"])).cpu().numpy()

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
                    ind_latents[i] = self.rep_model.get_encoding(x.to(self.config["device"])).cpu().numpy()
                    losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),y.to(self.config["device"])).cpu().numpy()
            pkl.dump(ind_latents, open(fname_encodings, "wb"))
            pkl.dump(losses, open(fname_losses, "wb"))

        ood_latents = np.zeros((len(ood_dataset), self.rep_model.latent_dim))
        ood_losses = np.zeros(len(ood_dataset))
        for i, (x, y, _) in tqdm(enumerate(DataLoader(ood_dataset, sampler=ood_sampler)), total=len(ood_dataset)):
            with torch.no_grad():
                ood_latents[i] = self.rep_model.get_encoding(x.to(self.config["device"])).cpu().numpy()

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