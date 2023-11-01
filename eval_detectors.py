# from yellowbrick.features import PCA
from testbeds import CIFAR10TestBed, NicoTestBed
import torch
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


def compute_stats(ind_pvalues, ood_pvalues_fold, ind_sample_losses, ood_sample_losses_fold, fname):
    df = convert_to_pandas_df(ind_pvalues, ood_pvalues_fold, ind_sample_losses, ood_sample_losses_fold)
    df.to_csv(fname)

def collect_data(sample_range, testbed_constructor, dataset_name):

    for sample_size in sample_range:
        testbed = testbed_constructor(sample_size, "classifier")
        tsd = RabanserSD(bench.classifier, select_samples=True,k=5, processes=1)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"), fname=f"data/{dataset_name}_ks_5NN_{sample_size}_fullloss.csv")

    for sample_size in sample_range:
        testbed = testbed_constructor(sample_size, "classifier")
        tsd = RabanserSD(bench.classifier, processes=1)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"),
                      fname=f"data/{dataset_name}_ks_{sample_size}_fullloss.csv")

    for sample_size in sample_range:
        testbed = testbed_constructor(sample_size, "classifier")
        tsd = TypicalitySD(bench.vae)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size),
                      fname=f"data/{dataset_name}_ks_{sample_size}_fullloss.csv")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    sample_range = [50, 100, 200, 500]


    for sample_size in [100]:
        # bench = NicoTestBed(sample_size)
        bench =CIFAR10TestBed(sample_size, rep_model="classifier", mode="severity")
        tsd = RabanserSD(bench.classifier, processes=1)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size, "ks"), fname=f"data/CIAR10_ks_{sample_size}_severity.csv")

    for sample_size in [100]:
        # bench = NicoTestBed(sample_size)
        bench =CIFAR10TestBed(sample_size, rep_model="classifier", mode="severity")
        tsd = RabanserSD(bench.classifier, select_samples=True, processes=1)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size, "ks"), fname=f"data/CIFAR10_ks_5NN_{sample_size}_severity.csv")

    for sample_size in [100]:
        # bench = NicoTestBed(sample_size)
        bench = CIFAR10TestBed(sample_size, rep_model="vae", mode="severity")
        tsd = TypicalitySD(bench.vae)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size), fname=f"data/CIFAR10_typicality_{sample_size}_severity.csv")

    # for sample_size in sample_range:
    #     # bench = NjordTestBed(sample_size)
    #     bench = NicoTestBed(sample_size, rep_model="vae")
    #     tsd = TypicalitySD(bench.vae)
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size),
    #                   fname=f"data/Nico_Typicality_{sample_size}_fullloss.csv")

    for sample_size in sample_range:
        # bench = NjordTestBed(sample_size)
        bench = CIFAR100TestBed(sample_size, rep_model="classifier")
        tsd = RabanserSD(bench.classifier, processes=2)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"),
                      fname=f"data/CIFAR100_ks_{sample_size}_fullloss.csv")

    for sample_size in sample_range:
        # bench = NjordTestBed(sample_size)
        bench = CIFAR100TestBed(sample_size, rep_model="classifier")
        tsd = RabanserSD(bench.classifier, select_samples=True, processes=2)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"),
                      fname=f"data/CIFAR100_ks_5NN_{sample_size}_fullloss.csv")

    # for sample_size in sample_range:
    #     # bench = NicoTestBed(sample_size)
    #     bench = NicoTestBed(sample_size, rep_model="vae")
    #     tsd = TypicalitySD(bench.vae)
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size), fname=f"data/NICO_typicality_{sample_size}_fullloss.csv")


    # for sample_size in sample_range:
    #     # bench = NicoTestBed(sample_size)
    #     bench = PolypTestBed(sample_size, rep_model="vae")
    #     tsd = TypicalitySD(bench.vae)
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size), fname=f"data/Polyp_typicality_{sample_size}_fullloss.csv")

    # for sample_size in sample_range:
    #     # bench = NicoTestBed(sample_size)
    #     # bench = PolypTestBed(sample_size, rep_model="classifier")
    #     bench = NjordTestBed(sample_size)
    #     tsd = RabanserSD(bench.vae)
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"), fname=f"data/Njord_ks_{sample_size}_fullloss.csv")
    #
    # for sample_size in sample_range:
    #     # bench = NicoTestBed(sample_size)
    #     # bench = PolypTestBed(sample_size, rep_model="classifier")
    #     bench = NjordTestBed(sample_size)
    #     # tsd = RabanserSD(bench.classifier, select_samples=True, k=5)
    #     tsd = RabanserSD(bench.vae, select_samples=True, k=5)
    #
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"), fname=f"data/Njord_ks_5NN_{sample_size}_fullloss.csv")
    #
    #
    # for sample_size in sample_range:
    #     # bench = NicoTestBed(sample_size)
    #     bench = NjordTestBed(sample_size)
    #     # bench = PolypTestBed(sample_size, rep_model="classifier")
    #     # tsd = RabanserSD(bench.classifier, select_samples=True, k=1)
    #     tsd = RabanserSD(bench.vae, select_samples=True, k=1)
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size), fname=f"data/Njord_ks_1NN_{sample_size}_fullloss.csv")
    #
    # # for sample_size in sample_range:
    # #     bench = CIFAR100TestBed(sample_size, rep_model="vae")
    #     tsd = TypicalitySD(bench.vae)
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size), fname=f"data/NICO_typicality_{sample_size}_fullloss.csv")
