import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import torch.nn
import multiprocessing
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA as sklearnPCA
import metrics
from bias_samplers import *
from utils import *
from features import grad_magnitude
from tqdm import tqdm
import pickle as pkl
from domain_datasets import *
from vae.models.vanilla_vae import VanillaVAE
# import torch_two_sample as tts
from sklearn.decomposition import PCA


def process_dataframe(data, filter_noise=False, combine_losses=True, filter_by_sampler=""):
    # data = data[data["sampler"] != "ClassOrderSampler"]
    # print(pd.unique(data["sampler"]))
    if filter_by_sampler!="":
        data = data[data["sampler"]==filter_by_sampler]
    if "noise" in str(pd.unique(data["fold"])) and filter_noise:
        data = data[(data["fold"] == "noise_0.2") | (data["fold"] == "ind")]
    if isinstance(data["loss"], str):
        data["loss"] = data["loss"].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])
        if combine_losses:
            data["loss"] = data["loss"].apply(lambda x: np.mean(x))
        else:
            data=data.explode("loss")
    data["oodness"] = data["loss"] / data[data["fold"] == "ind"]["loss"].quantile(0.95)
    return data

def convert_to_pandas_df(ind_pvalues, ood_pvalues, ind_sample_losses, ood_sample_losses):
    dataset = []
    for fold, by_sampler in ind_pvalues.items():
        for sampler, data in by_sampler.items():
            dataset.append({"fold": fold, "sampler": sampler, "pvalue": ind_pvalues[fold][sampler], "loss": ind_sample_losses[fold][sampler]})

    for fold, by_sampler in ood_pvalues.items():
        for sampler, data in by_sampler.items():
            dataset.append({"fold": fold, "sampler": sampler, "pvalue": ood_pvalues[fold][sampler], "loss": ood_sample_losses[fold][sampler]})
    pkl.dump(dataset, open("data_nico_debug.pkl", "wb"))
    df = pd.DataFrame(dataset)

    df = df.explode(["pvalue", "loss"])
    return df

def get_debiased_samples(ind_encodings, ind_predictions, ind_features, sample_encodings, sample_predictions, sample_features, k=5, debias_labels=False):
    """
        Returns debiased features from the ind set.
    """

    k_nearest_idx = np.concatenate(
        [np.argpartition(
            torch.sum((torch.Tensor(sample_encodings[i]).unsqueeze(0) - ind_encodings) ** 2, dim=-1).numpy(), k)[
         :k] for i in
         range(len(sample_encodings))])

    k_nearest_ind = ind_features[k_nearest_idx]
    if debias_labels:
        k_nearest_labels = np.concatenate(
            [np.argpartition(
                torch.sum((torch.Tensor(sample_predictions[i]).unsqueeze(0) - ind_predictions) ** 2, dim=-1).numpy(), k)[
             :k] for i in
             range(len(sample_encodings))])
        nearest_labels = ind_features[k_nearest_labels]
        k_nearest_ind = np.concatenate([k_nearest_ind, nearest_labels], axis=-1)
    return k_nearest_ind
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
        print(encodings.shape)
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
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
                pass
                # mmd = tts.MMDStatistic(len(ind_encodings), sample_size)
                # value, matrix = mmd(ind_encodings,
                #                     ood_samples, alphas=[0.5], ret_matrix=True)
                # p_value = mmd.pval(matrix, n_permutations=100)
            elif test == "knn":
                pass
                # knn = tts.KNNStatistic(ind_encodings, sample_size, k=sample_size)
                # value, matrix = knn(ind_encodings, ood_samples,
                #                     ret_matrix=True)
                # p_value = knn.pval(matrix, n_permutations=100)
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
        print("encoded")
        losses =  dict(
            zip(dataloaders.keys(),
                [dict(zip(loader_w_sampler.keys(),
                         [self.testbed.compute_losses(loader)
                          for sampler_name, loader in loader_w_sampler.items()]
                         )) for
                     loader_w_sampler in dataloaders.values()]))
        print("losses computed)")

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

        # mmd = tts.MMDStatistic(len(ind_encodings), sample_size)
        # knn = tts.KNNStatistic(len(ind_encodings),sample_size, k=sample_size)
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
        print("got latents")
        ind_pvalues, ind_losses = self.compute_pvals_and_loss_for_loader(ind_latents, self.testbed.ind_val_loaders(), sample_size, test)
        print("got ind")

        ood_pvalues, ood_losses = self.compute_pvals_and_loss_for_loader(ind_latents, self.testbed.ood_loaders(), sample_size, test)
        print("got ood")
        return ind_pvalues, ood_pvalues, ind_losses, ood_losses

class TypicalitySD(BaseSD):
    def __init__(self, rep_model):
        super().__init__(rep_model)

    def compute_entropy(self, data_loader):
        log_likelihoods = []
        for i, batch in enumerate(data_loader):
            x = batch[0].to("cuda")
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

class FeatureSD(BaseSD):
    """
    General class for gradient-based detectors, including jacobian.
    Computes a gradient norm/jacobian norm/hessian norm/etc
    """
    def __init__(self, rep_model, feature_fn=grad_magnitude, num_features=1, k=0):
        super().__init__(rep_model)
        self.k=k
        self.feature_fn = feature_fn
        print("init:", self.feature_fn)
        self.num_features=num_features

    def get_features(self, dataloader):
        features = torch.zeros((len(dataloader), 1))
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = data[0].cuda()
            features[i] = self.feature_fn(self.rep_model, x, self.num_features)
        return features
    def get_features_encodings(self, dataloader):
        features = np.zeros((len(dataloader), 1))
        encodings = np.zeros((len(dataloader), self.rep_model.latent_dim))
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = data[0].cuda()
            with torch.no_grad():
                encodings[i] = self.rep_model.get_encoding(x).cpu().numpy()
            features[i] = self.feature_fn(self.rep_model, x, self.num_features)
        return features, encodings

    def paralell_process(self,start, stop, biased_sampler_encodings, biased_sampler_features, ind_encodings, ind_features, fold_name, biased_sampler_name, losses):
        sample_norms = biased_sampler_features[start:stop]
        sample_encodings = biased_sampler_encodings[start:stop]

        if self.k!=0:

            k_nearest_idx = np.concatenate(
                [np.argpartition(
                    torch.sum((torch.Tensor(sample_encodings[i]).unsqueeze(0) - ind_encodings) ** 2, dim=-1).numpy(),
                    self.k)[
                 :self.k] for i in
                 range(len(sample_encodings))])
            k_nearest_ind = ind_features[k_nearest_idx]

            p_value = ks_2samp(k_nearest_ind[:, 0], sample_norms[:, 0])[1]
            print("\t\t", p_value)
        else:
            p_value = ks_2samp(sample_norms[:,0], ind_features[:,0])[1]
        return p_value,losses[fold_name][biased_sampler_name][start:stop]

    def compute_pvals_and_loss_for_loader(self, ind_norms, ind_encodings, dataloaders, sample_size):

        features_encodings = dict(
            zip(dataloaders.keys(),
                [dict(zip(loader_w_sampler.keys(),
                          [self.get_features_encodings(loader)
                           for sampler_name, loader in loader_w_sampler.items()]
                          )) for
                 loader_w_sampler in
                 dataloaders.values()]))

        # encodings = dict(
        #     zip(dataloaders.keys(),
        #         [dict(zip(loader_w_sampler.keys(),
        #                   [self.get_encodings(loader)
        #                    for sampler_name, loader in loader_w_sampler.items()]
        #                   )) for
        #          loader_w_sampler in
        #          dataloaders.values()]))

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

        for fold_name, fold_entropies in features_encodings.items():
            print(fold_name)
            for biased_sampler_name, biased_sampler_norms_encodings in fold_entropies.items():
                print("\t", biased_sampler_name)
                biased_sampler_norms, biased_sampler_encodings = biased_sampler_norms_encodings

                args = [ biased_sampler_encodings, biased_sampler_norms, ind_encodings, ind_norms, fold_name, biased_sampler_name, losses]
                pool = multiprocessing.Pool(processes=10)
                startstop_iterable = list(zip(range(0, len(biased_sampler_encodings), sample_size),
                                            range(sample_size, len(biased_sampler_encodings) + sample_size, sample_size)))[
                                   :-1]
                # results = []
                # for start, stop in startstop_iterable:
                #     results.append(self.paralell_process(start, stop, biased_sampler_encodings, biased_sampler_norms, ind_encodings, ind_norms, fold_name, biased_sampler_name, losses))
                results = pool.starmap(self.paralell_process, ArgumentIterator(startstop_iterable, args))
                pool.close()
                for p_value, sample_loss in results:
                    p_values[fold_name][biased_sampler_name].append(p_value)
                    sample_losses[fold_name][biased_sampler_name].append(sample_loss)
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

        ind_norms, ind_encodings = self.get_features_encodings(self.testbed.ind_loader())
        print(ind_encodings.shape)
        print(ind_norms.shape)
        # compute ind_val pvalues for each sampler
        ind_pvalues, ind_losses = self.compute_pvals_and_loss_for_loader(ind_norms, ind_encodings, self.testbed.ind_val_loaders(), sample_size)
        #todo: make this use less memory
        ood_pvalues, ood_losses = self.compute_pvals_and_loss_for_loader(ind_norms, ind_encodings, self.testbed.ood_loaders(), sample_size)
        return ind_pvalues, ood_pvalues, ind_losses, ood_losses




def open_and_process(fname, filter_noise=False, combine_losses=True, exclude_sampler=""):
    try:
        data = pd.read_csv(fname)
        # data = data[data["sampler"] != "ClassOrderSampler"]
        # print(pd.unique(data["sampler"]))
        if exclude_sampler!="":
            data = data[data["sampler"]!=exclude_sampler]
        if "noise" in str(pd.unique(data["fold"])) and filter_noise:
            max_noise = sorted([float(i.split("_")[1]) for i in pd.unique(data["fold"]) if "noise" in i])[-1]
            data = data[(data["fold"] == f"noise_{max_noise}") | (data["fold"] == "ind")]
        try:
            data["loss"] = data["loss"].map(lambda x: float(x))
        except:

            data["loss"] = data["loss"].str.strip('[]').str.split().apply(lambda x: [float(i) for i in x])
            if combine_losses:
                data["loss"] = data["loss"].apply(lambda x: np.mean(x))
            else:
                data=data.expbrlode("loss")
        data["oodness"] = data["loss"] / data[data["fold"] == "ind"]["loss"].quantile(0.95)


        return data
    except FileNotFoundError:
        print(f"File {fname} not found")
        return None
