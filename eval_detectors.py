import os
from torch.utils.data import ConcatDataset
import multiprocessing, logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metrics import *
from yellowbrick.features import PCA
from vae.vae_experiment import VAEXperiment
from vae.models.vanilla_vae import VanillaVAE, ResNetVAE
import yaml
import torch
from segmentor.deeplab import SegmentationModel
from torch.utils.data import DataLoader
from utils import *
from yellowbrick.features.manifold import Manifold
from sklearn.manifold import SpectralEmbedding, Isomap
from scipy.stats import ks_2samp
from bias_samplers import *
from torch.utils.data.sampler import RandomSampler
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import CIFAR10,CIFAR100,MNIST
import pickle as pkl
import torch.utils.data as data
from domain_datasets import *
from torch.utils.data import RandomSampler
from classifier.resnetclassifier import ResNetClassifier
from ooddetectors import *
from testbeds import *







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
def compute_stats(ind_pvalues, ood_pvalues_fold, ind_sample_losses, ood_sample_losses_fold, fname):
    df = convert_to_pandas_df(ind_pvalues, ood_pvalues_fold, ind_sample_losses, ood_sample_losses_fold)
    df.to_csv(fname)
    ind_pvalues_ready = list(ind_pvalues.values())[0]
    for fold in ood_pvalues_fold.keys():
        for sampler in ood_pvalues_fold[fold].keys():
            ood_pvalues = ood_pvalues_fold[fold][sampler]
            auroc = metrics.auroc(ood_pvalues, ind_pvalues_ready[sampler])
            fpr = metrics.fprat95tpr(ood_pvalues, ind_pvalues_ready[sampler])
            accuracy = metrics.calibrated_detection_rate(ood_pvalues, ind_pvalues_ready[sampler])
            print(f"{fold} : {sampler} AUROC: {auroc}")
            print(f"{fold} : {sampler} FPR: {fpr}")
            print(f"{fold} : {sampler} Accuracy: {accuracy}")
            print()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    for sample_size in [10, 20, 50, 100, 200, 500]:
        # bench = NicoTestBed(sample_size)
        bench =CIFAR10TestBed(sample_size)
        tsd = RabanserSD(bench.rep_model, None, select_samples=False)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"), fname=f"CIFAR_classifier_ks_{sample_size}_fullloss.csv")

    for sample_size in [10, 20, 50, 100, 200, 500]:
        bench =CIFAR10TestBed(sample_size)
        # bench = NicoTestBed(sample_size)
        tsd = RabanserSD(bench.rep_model, None, select_samples=True)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"), fname=f"CIFAR_classifier_ks_5NN_{sample_size}_fullloss.csv")