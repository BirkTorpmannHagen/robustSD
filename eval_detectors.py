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



if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    #CIFAR100
    # for sample_size in [10, 20, 50, 100, 200, 500][::-1]:
    #     bench = CIFAR100TestBed(sample_size)
    #     tsd = TypicalitySD(bench.vae)
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size), fname=f"CIFAR100_classifier_typicality_{sample_size}_fullloss.csv")

    # for sample_size in [10, 20, 50, 100, 200, 500][::-1]:
    #     # bench =CIFAR10TestBed(sample_size)
    #     bench = CIFAR100TestBed(sample_size)
    #     tsd = RabanserSD(bench.rep_model, select_samples=False)
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"), fname=f"CIFAR100_classifier_ks_{sample_size}_fullloss.csv")

    # for sample_size in [10, 20, 50, 100, 200, 500][::-1]:
    #     bench = CIFAR100TestBed(sample_size)
    #     tsd = RabanserSD(bench.rep_model, select_samples=True)
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"),
    #                   fname=f"CIFAR100_classifier_ks_5NN_{sample_size}_fullloss.csv")

    # for sample_size in [10, 20, 50, 100, 200, 500][::-1]:
    #     bench = NicoTestBed(sample_size)
    #     tsd = RabanserSD(bench.rep_model, select_samples=False)
    #     tsd.register_testbed(bench)
    #     compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"), fname=f"NICO_ks_{sample_size}_fullloss.csv")

    for sample_size in [10, 20][::-1]:
        bench = ImagenetteTestBed(sample_size)
        tsd = RabanserSD(bench.rep_model, select_samples=True)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"), fname=f"data/imagenette_ks_5NN_{sample_size}_fullloss.csv")
