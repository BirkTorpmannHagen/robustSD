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
# import torch_two_sample as tts
from sklearn.decomposition import PCA

class BaseSD:
    def __init__(self, rep_model):
        self.rep_model = rep_model

    def register_testbed(self, testbed):
        self.testbed = testbed


class RabanserSD(BaseSD):
    def __init__(self, rep_model,  select_samples=False, k=5, processes=5):
        super().__init__(rep_model)
        self.select_samples = select_samples
        self.k= k
        self.processes = processes
        # if set_start:
        #     torch.multiprocessing.set_start_method('spawn') #bodge code, sorry.

    def get_encodings(self, dataloader):
        encodings = np.zeros((len(dataloader), self.rep_model.latent_dim))
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

                #samples x
                p_value = np.min([ks_2samp(k_nearest_ind[:, i], ood_samples[:, i])[-1] for i in
                                  range(self.rep_model.latent_dim)])
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
        return p_value, losses[fold_name][biased_sampler_name][start:stop]

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
        for fold_name, fold_encodings in encodings.items():
            for biased_sampler_name, biased_sampler_encodings in fold_encodings.items():
                ind_encodings = torch.Tensor(ind_encodings)
                biased_sampler_encodings = torch.Tensor(biased_sampler_encodings)

                args = [   biased_sampler_encodings, ind_encodings, test, fold_name, biased_sampler_name, losses, sample_size]
                pool = multiprocessing.Pool(processes=self.processes)

                startstop_iterable = list(zip(range(0, len(biased_sampler_encodings), sample_size),
                                            range(sample_size, len(biased_sampler_encodings) + sample_size, sample_size)))[
                                   :-1]
                results = pool.starmap(self.paralell_process, ArgumentIterator(startstop_iterable, args))
                pool.close()
                # results = []
                # for start, stop in list(zip(range(0, len(biased_sampler_encodings), sample_size),
                #                             range(sample_size, len(biased_sampler_encodings) + sample_size, sample_size)))[
                #                    :-1]:
                #     results.append(self.paralell_process(start, stop, biased_sampler_encodings, ind_encodings, test, fold_name, biased_sampler_name, losses, sample_size))

                # results = map(self.paralell_process,ArgumentIterator(startstop_iterable, args))

                # print(f"the results for {fold_name}:{biased_sampler_name} are {results} ")
                # input()
                for p_value, sample_loss in results:

                    p_values[fold_name][biased_sampler_name].append(p_value)
                    sample_losses[fold_name][biased_sampler_name].append(sample_loss)

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

class KNNDSD(RabanserSD):

    def paralell_process(self, start, stop, biased_sampler_encodings, ind_encodings, test, fold_name, biased_sampler_name, losses, sample_size):
        # biased_sampler_encodings, ind_encodings, test, fold_name, biased_sampler_name, losses, sample_size = args
        ood_samples = biased_sampler_encodings[start:stop]

        k_nearest_dists= np.concatenate(
            [np.partition(torch.sum((ood_samples[i].unsqueeze(0)- ind_encodings) ** 2, dim=-1).numpy(), self.k)[:self.k] for i in
             range(len(ood_samples))])

        dist = np.mean(k_nearest_dists, axis=1)
        print(dist)
        p_value = 1 - np.mean([1 if dist > i else 0 for i in self.ind_dist_distribution])

        def compute_pvals_and_loss_for_loader(self, ind_encodings, dataloaders, sample_size, test):

            encodings = dict(
                zip(dataloaders.keys(),
                    [dict(zip(loader_w_sampler.keys(),
                              [self.get_encodings(loader)
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

            mmd = tts.MMDStatistic(len(ind_encodings), sample_size)
            knn = tts.KNNStatistic(len(ind_encodings), sample_size, k=sample_size)
            print(losses)
            for fold_name, fold_encodings in encodings.items():
                for biased_sampler_name, biased_sampler_encodings in fold_encodings.items():
                    ind_encodings = torch.Tensor(ind_encodings)
                    biased_sampler_encodings = torch.Tensor(biased_sampler_encodings)

                    args = [biased_sampler_encodings, ind_encodings, test, fold_name, biased_sampler_name, losses,
                            sample_size]
                    pool = multiprocessing.Pool(processes=4)

                    startstop_iterable = list(zip(range(0, len(biased_sampler_encodings), sample_size),
                                                  range(sample_size, len(biased_sampler_encodings) + sample_size,
                                                        sample_size)))[
                                         :-1]
                    # results = pool.starmap(self.paralell_process, ArgumentIterator(startstop_iterable, args))
                    # pool.close()
                    # results = []
                    # for start, stop in list(zip(range(0, len(biased_sampler_encodings), sample_size),
                    #                             range(sample_size, len(biased_sampler_encodings) + sample_size,
                    #                                   sample_size)))[
                    #                    :-1]:
                    #     results.append(
                    #         self.paralell_process(start, stop, biased_sampler_encodings, ind_encodings, test, fold_name,
                    #                               biased_sampler_name, losses, sample_size))

                    results = map(self.paralell_process,ArgumentIterator(startstop_iterable, args))

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

            ind_pvalues, ind_losses = self.compute_pvals_and_loss_for_loader(ind_latents,
                                                                             self.testbed.ind_val_loaders(),
                                                                             sample_size, test)
            ood_pvalues, ood_losses = self.compute_pvals_and_loss_for_loader(ind_latents, self.testbed.ood_loaders(),
                                                                             sample_size, test)
            return ind_pvalues, ood_pvalues, ind_losses, ood_losses


                # if fold_name == "ind":
class TypicalitySD(BaseSD):
    def __init__(self, rep_model):
        super().__init__(rep_model)

    def compute_entropy(self, data_loader):
        print(data_loader.sampler)
        print(data_loader)
        log_likelihoods = []
        for i, (x, y) in enumerate(data_loader):
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
                    sample_losses[fold_name][biased_sampler_name].append(
                        losses[fold_name][biased_sampler_name][start:stop])

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

        ind_entropies = self.compute_entropy(self.testbed.ind_loader())
        bootstrap_entropy_distribution = sorted(
            [np.random.choice(ind_entropies, sample_size).mean().item() for i in range(10000)])
        entropy_epsilon = np.quantile(bootstrap_entropy_distribution, 0.99)  # alpha of .99 quantile

        # compute ind_val pvalues for each sampler
        ind_pvalues, ind_losses = self.compute_pvals_and_loss_for_loader(bootstrap_entropy_distribution,
                                                                         self.testbed.ind_val_loaders(), sample_size)


        ood_pvalues, ood_losses = self.compute_pvals_and_loss_for_loader(bootstrap_entropy_distribution,
                                                                         self.testbed.ood_loaders(), sample_size)
        return ind_pvalues, ood_pvalues, ind_losses, ood_losses
