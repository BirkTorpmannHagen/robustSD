import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn
import multiprocessing
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA as sklearnPCA
import metrics
from bias_samplers import *
from utils import *
from tqdm import tqdm
import pickle as pkl
from domain_datasets import *
from vae.models.vanilla_vae import VanillaVAE
import torch_two_sample as tts
from sklearn.decomposition import PCA

class BaseSD:
    def __init__(self, rep_model, sample_selector):
        self.sample_selector = sample_selector
        self.rep_model = rep_model

    def register_testbed(self, testbed):
        self.testbed = testbed


class RabanserSD(BaseSD):
    def __init__(self, rep_model, sample_selector, select_samples=False, k=5):
        super().__init__(rep_model, sample_selector)
        self.select_samples = select_samples
        self.k= k
        # if set_start:
        #     torch.multiprocessing.set_start_method('spawn') #bodge code, sorry.

    def get_encodings(self, dataloader):
        encodings = np.zeros((len(dataloader), self.rep_model.latent_dim))
        print(encodings.shape)
        for i, data in enumerate(dataloader):
            x = data[0]
            with torch.no_grad():
                x = x.to("cuda").float()
                encodings[i] = self.rep_model.get_encoding(x).cpu().numpy() #mu from vae or features from classifier
        return encodings

    def get_k_nearest(self, ind_samples):
        pass


    def paralell_process(self, start, stop, biased_sampler_encodings, ind_encodings, test, fold_name, biased_sampler_name, losses, sample_size):
        # biased_sampler_encodings, ind_encodings, test, fold_name, biased_sampler_name, losses, sample_size = args
        ood_samples = biased_sampler_encodings[start:stop]
        if test == "ks":
            if self.select_samples:
                #ood x k
                k_nearest_idx = np.concatenate(
                    [np.argpartition(torch.sum((ood_samples[i].unsqueeze(0)- ind_encodings) ** 2, dim=-1).numpy(), self.k)[:self.k] for i in
                     range(len(ood_samples))])
                k_nearest_ind = ind_encodings[k_nearest_idx]
                # pca = PCA()
                # ind_transformed = pca.fit_transform(ind_encodings)
                # nn_trans = pca.transform(k_nearest_ind)
                # nn_ood = pca.transform(ood_samples)

                #samples x
                p_value = np.min([ks_2samp(k_nearest_ind[:, i], ood_samples[:, i])[-1] for i in
                                  range(self.rep_model.latent_dim)])
                # if fold_name == "ind":
                #     plt.scatter(ind_transformed[:, 0], ind_transformed[:, 1], alpha=0.5, label="ind")
                #     plt.scatter(nn_trans[:, 0], nn_trans[:, 1], alpha=0.5, label="nearest neighbours")
                #     plt.scatter(nn_ood[:, 0], nn_ood[:, 1], alpha=0.5, label="sample")
                #     plt.legend()
                #     plt.savefig(f"test_{start}.png")
                #     plt.title(p_value)
                #     plt.clf()
                #     plt.close()
            else:
                p_value = np.min([ks_2samp(ind_encodings[:, i], ood_samples[:, i])[-1] for i in
                              range(self.rep_model.latent_dim)])
        else:
            if test == "mmd":
                mmd = tts.MMDStatistic(len(ind_encodings), sample_size)
                value, matrix = mmd(ind_encodings,
                                    ood_samples, alphas=[0.5], ret_matrix=True)
                p_value = mmd.pval(matrix, n_permutations=100)
            elif test == "knn":
                knn = tts.KNNStatistic(ind_encodings, sample_size, k=sample_size)
                value, matrix = knn(ind_encodings, ood_samples,
                                    ret_matrix=True)
                p_value = knn.pval(matrix, n_permutations=100)
            else:
                raise NotImplementedError
        return p_value, losses[fold_name][biased_sampler_name][start:stop].mean()

    def compute_pvals_and_loss_for_loader(self,ind_encodings, dataloaders, sample_size, test):



        encodings = dict(
            zip(dataloaders.keys(),
                [dict(zip(loader_w_sampler.keys(),
                         [self.get_encodings(loader)
                          for sampler_name, loader in loader_w_sampler.items()]
                         )) for
                     loader_w_sampler in dataloaders.values()])) #dict of dicts of tensors; sidenote initializing nested dicts sucks

        losses =  dict(
            zip(dataloaders.keys(),
                [dict(zip(loader_w_sampler.keys(),
                         [self.testbed.compute_losses(loader)
                          for sampler_name, loader in loader_w_sampler.items()]
                         )) for
                     loader_w_sampler in dataloaders.values()]))

        p_values = dict(
            zip(dataloaders.keys(),
                [dict(zip(loader_w_sampler.keys(),
                         [[]
                          for _ in range(len(loader_w_sampler))]
                         )) for
                     loader_w_sampler in dataloaders.values()]))

        sample_losses = dict(
            zip(dataloaders.keys(),
                [dict(zip(loader_w_sampler.keys(),
                         [[]
                          for _ in range(len(loader_w_sampler))]
                         )) for
                     loader_w_sampler in dataloaders.values()]))

        mmd = tts.MMDStatistic(len(ind_encodings), sample_size)
        knn = tts.KNNStatistic(len(ind_encodings),sample_size, k=sample_size)
        print(losses)
        for fold_name, fold_encodings in encodings.items():
            for biased_sampler_name, biased_sampler_encodings in fold_encodings.items():
                ind_encodings = torch.Tensor(ind_encodings)
                biased_sampler_encodings = torch.Tensor(biased_sampler_encodings)

                args = [   biased_sampler_encodings, ind_encodings, test, fold_name, biased_sampler_name, losses, sample_size]
                # pool = multiprocessing.Pool(processes=4)

                startstop_iterable = list(zip(range(0, len(biased_sampler_encodings), sample_size),
                                            range(sample_size, len(biased_sampler_encodings) + sample_size, sample_size)))[
                                   :-1]
                # results = pool.starmap(self.paralell_process, ArgumentIterator(startstop_iterable, args))
                # pool.close()
                results = []
                for start, stop in list(zip(range(0, len(biased_sampler_encodings), sample_size),
                                            range(sample_size, len(biased_sampler_encodings) + sample_size, sample_size)))[
                                   :-1]:
                    results.append(self.paralell_process(start, stop, biased_sampler_encodings, ind_encodings, test, fold_name, biased_sampler_name, losses, sample_size))

                # results = map(self.paralell_process,ArgumentIterator(startstop_iterable, args))

                print(f"the results for {fold_name}:{biased_sampler_name} are {results} ")
                # input()
                for p_value, sample_loss in results:

                    p_values[fold_name][biased_sampler_name].append(p_value)
                    sample_losses[fold_name][biased_sampler_name].append(sample_loss.item())

        return p_values, sample_losses


    def compute_pvals_and_loss(self, sample_size, test):
        """

        :param sample_size: sample size for the tests
        :return: ind_p_values: p-values for ind fold for each sampler
        :return ood_p_values: p-values for ood fold for each sampler
        :return ind_sample_losses: losses for each sampler on ind fold, in correct order
        :return ood_sample_losses: losses for each sampler on ood fold, in correct order
        """
        # sample_size = min(sample_size, len(self.testbed.ind_val_loaders()[0]))
        # try:
        #     ind_latents = torch.load(f"{type(self).__name__}_{type(self.testbed).__name__}.pt")
        # except FileNotFoundError:
        #     print("recomputing...")
        ind_latents = self.get_encodings(self.testbed.ind_loader())
        torch.save(ind_latents, f"{type(self).__name__}_{type(self.testbed).__name__}.pt")

        ind_pvalues, ind_losses = self.compute_pvals_and_loss_for_loader(ind_latents, self.testbed.ind_val_loaders(), sample_size, test)
        ood_pvalues, ood_losses = self.compute_pvals_and_loss_for_loader(ind_latents, self.testbed.ood_loaders(), sample_size, test)
        return ind_pvalues, ood_pvalues, ind_losses, ood_losses


class TypicalitySD(BaseSD):
    def __init__(self, rep_model, sample_selector):
        super().__init__(rep_model, sample_selector)

    def compute_entropy(self, data_loader):
        log_likelihoods = []
        for i, (x, y, _) in enumerate(data_loader):
            x = x.to("cuda")
            log_likelihoods.append(self.rep_model.estimate_log_likelihood(x))
        entropies = -(torch.tensor(log_likelihoods))
        return entropies

    def compute_pvals_and_loss_for_loader(self, bootstrap_entropy_distribution, dataloaders, sample_size):
        entropies = dict(
            zip(dataloaders.keys(),
                [dict(zip(loader_w_sampler.keys(),
                          [self.compute_entropy(loader)
                           for sampler_name, loader in loader_w_sampler.items()]
                          )) for
                 loader_w_sampler in
                 dataloaders.values()]))  # dict of dicts of tensors; sidenote initializing nested dicts sucks

        losses = dict(
            zip(dataloaders.keys(),
                [dict(zip(loader_w_sampler.keys(),
                          [self.testbed.compute_losses(loader)
                           for sampler_name, loader in loader_w_sampler.items()]
                          )) for
                 loader_w_sampler in dataloaders.values()]))

        p_values = dict(
            zip(dataloaders.keys(),
                [dict(zip(loader_w_sampler.keys(),
                          [[]
                           for _ in range(len(loader_w_sampler))]
                          )) for
                 loader_w_sampler in dataloaders.values()]))

        sample_losses = dict(
            zip(dataloaders.keys(),
                [dict(zip(loader_w_sampler.keys(),
                          [[]
                           for _ in range(len(loader_w_sampler))]
                          )) for
                 loader_w_sampler in dataloaders.values()]))

        for fold_name, fold_entropies in entropies.items():
            for biased_sampler_name, biased_sampler_entropies in fold_entropies.items():
                for start, stop in list(zip(range(0, len(biased_sampler_entropies), sample_size),
                                            range(sample_size, len(biased_sampler_entropies) + sample_size,
                                                  sample_size)))[
                                   :-1]:
                    sample_entropy = torch.mean(biased_sampler_entropies[start:stop])
                    p_value = 1 - np.mean([1 if sample_entropy > i else 0 for i in bootstrap_entropy_distribution])

                    p_values[fold_name][biased_sampler_name].append(p_value)
                    sample_losses[fold_name][biased_sampler_name].append(np.mean(
                        losses[fold_name][biased_sampler_name][start:stop]))

        return p_values, sample_losses

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
            for x, y, _ in self.testbed.ind_loader():
                x = x.to("cuda")
                loglikelihoods.append(self.rep_model.estimate_log_likelihood(x))
            torch.save(loglikelihoods, f"{type(self).__name__}_{type(self.testbed).__name__}.pt")
        resub_entropy = -(torch.tensor(loglikelihoods)).mean().item()
        ind_entropies = -(torch.tensor(loglikelihoods))
        bootstrap_entropy_distribution = sorted(
            [np.random.choice(ind_entropies, sample_size).mean().item() for i in range(10000)])
        entropy_epsilon = np.quantile(bootstrap_entropy_distribution, 0.99)  # alpha of .99 quantile

        # compute ind_val pvalues for each sampler
        ind_pvalues, ind_losses = self.compute_pvals_and_loss_for_loader(bootstrap_entropy_distribution,
                                                                         self.testbed.ind_val_loaders(), sample_size)
        ood_pvalues, ood_losses = self.compute_pvals_and_loss_for_loader(bootstrap_entropy_distribution,
                                                                         self.testbed.ood_loaders(), sample_size)
        return ind_pvalues, ood_pvalues, ind_losses, ood_losses


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