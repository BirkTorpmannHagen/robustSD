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


def collect_gradient_data(sample_range, testbed_constructor, dataset_name, grad_fn, mode="normal", k=0):
    print(grad_fn)
    for sample_size in sample_range:
        if grad_fn==typicality:
            bench = testbed_constructor(sample_size, "vae", mode=mode)
            tsd = FeatureSD(bench.vae, grad_fn, k=k)
        else:
            bench = testbed_constructor(sample_size, "classifier", mode=mode)
            tsd = FeatureSD(bench.classifier, grad_fn,k=k)
        tsd.register_testbed(bench)
        if k!=0:
            name = f"new_data/{dataset_name}_{mode}_{grad_fn.__name__}_{k}NN_{sample_size}.csv"
        else:
            name = f"new_data/{dataset_name}_{mode}_{grad_fn.__name__}_{sample_size}.csv"
        compute_stats(*tsd.compute_pvals_and_loss(sample_size),
                      fname=name)

def collect_data(sample_range, testbed_constructor, dataset_name, mode="normal"):
    if testbed_constructor==NjordTestBed:
        for sample_size in sample_range:
            bench = testbed_constructor(sample_size, mode=mode)
            tsd = RabanserSD(bench.vae, select_samples=True, k=5, processes=1)
            tsd.register_testbed(bench)
            compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"),
                          fname=f"new_data/{dataset_name}_{mode}_ks_5NN_{sample_size}.csv")

        for sample_size in sample_range:
            bench = testbed_constructor(sample_size, mode=mode)
            tsd = RabanserSD(bench.vae, processes=1)
            tsd.register_testbed(bench)
            compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"),
                          fname=f"new_data/{dataset_name}_{mode}_ks_{sample_size}.csv")
        for sample_size in sample_range:
            bench = testbed_constructor(sample_size, mode=mode)
            tsd = TypicalitySD(bench.vae)
            tsd.register_testbed(bench)
            compute_stats(*tsd.compute_pvals_and_loss(sample_size),
                          fname=f"new_data/{dataset_name}_{mode}_typicality_{sample_size}.csv")
    else:
        # for sample_size in sample_range:
        #     bench = testbed_constructor(sample_size, "vae", mode=mode)
        #     tsd = TypicalitySD(bench.vae)
        #     tsd.register_testbed(bench)
        #     compute_stats(*tsd.compute_pvals_and_loss(sample_size),
        #                   fname=f"new_data/{dataset_name}_{mode}_typicality_{sample_size}.csv")
        for sample_size in sample_range:
            bench = testbed_constructor(sample_size, "classifier", mode=mode)
            tsd = FeatureSD(bench.classifier, processes=5)
            tsd.register_testbed(bench)
            compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"),
                          fname=f"new_data/{dataset_name}_{mode}_gradient_{sample_size}.csv")


        # for sample_size in sample_range:
        #     bench = testbed_constructor(sample_size, "classifier", mode=mode)
        #     tsd = RabanserSD(bench.classifier, select_samples=True,k=5, processes=1)
        #     tsd.register_testbed(bench)
        #     compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"),
        #                   fname=f"new_data/{dataset_name}_{mode}_ks_5NN_{sample_size}.csv")
        #
        # for sample_size in sample_range:
        #     bench = testbed_constructor(sample_size, "classifier", mode=mode)
        #     tsd = RabanserSD(bench.classifier, processes=1)
        #     tsd.register_testbed(bench)
        #     compute_stats(*tsd.compute_pvals_and_loss(sample_size, test="ks"),
        #                   fname=f"new_data/{dataset_name}_{mode}_ks_{sample_size}.csv")


def collect_severitydata():
    # collect_data([100], CIFAR10TestBed, "CIFAR10", mode="severity")
    collect_data([100], CIFAR100TestBed, "CIFAR100", mode="severity")
    collect_data([100], ImagenetteTestBed, "imagenette", mode="severity")
    collect_data([100], NicoTestBed, "NICO", mode="severity")

    # collect_data(sample_range, PolypTestBed, "Polyp")
    # collect_data(sample_range, NjordTestBed, "Njord")

def collect_all_data(sample_range):
    pass
    # collect_data(sample_range, PolypTestBed, "Polyp", mode="noise")
    # collect_data(sample_range, NicoTestBed, "NICO", mode="noise")
    # collect_data(sample_range, NjordTestBed, "Njord", mode="noise")

    collect_data(sample_range, CIFAR10TestBed, "CIFAR10")
    collect_data(sample_range, CIFAR100TestBed, "CIFAR100")
    collect_data(sample_range, ImagenetteTestBed, "imagenette")
    # collect_data(sample_range, PolypTestBed, "Polyp")
    #collect_data(sample_range, NicoTestBed, "NICO")
    # collect_data(sample_range, NjordTestBed, "Njord")

    # collect_data(sample_range, CIFAR10TestBed, "CIFAR10", mode="severity")
    # collect_data(sample_range, CIFAR100TestBed, "CIFAR100", mode="severity")
    # collect_data(sample_range, ImagenetteTestBed, "imagenette", mode="severity")
    # collect_data(sample_range, PolypTestBed, "Polyp", mode="severity")
    # collect_data(sample_range, NicoTestBed, "NICO", mode="severity")
    # collect_data(sample_range, NjordTestBed, "Njord", mode="severity")

def grad_data(k=0):
    sample_range = [10, 30]

    for grad_fn in [odin, cross_entropy, grad_magnitude]:
        for k in [0, 5]:
            collect_gradient_data(sample_range, NicoTestBed, "NICO", grad_fn, k=k)
            collect_gradient_data(sample_range, ImagenetteTestBed, "imagenette", grad_fn, k=k)
            if k==0 and grad_fn == odin:
                continue
            collect_gradient_data(sample_range, CIFAR10TestBed, "CIFAR10", grad_fn, k=k)
            collect_gradient_data(sample_range, CIFAR100TestBed, "CIFAR100", grad_fn, k=k)


        # collect_gradient_data(sample_range, NjordTestBed, "NICO", grad_fn,k=k)


  #  sample_range = [100]
    # for grad_fn in [condition_number]:
    #     collect_gradient_data(sample_range, CIFAR10TestBed, "CIFAR10", grad_fn)
    #     collect_gradient_data(sample_range, CIFAR100TestBed, "CIFAR100", grad_fn)
    #     collect_gradient_data(sample_range, NicoTestBed, "NICO", grad_fn)
  #  for grad_fn in [jacobian, condition_number]:
    #     collect_gradient_data(sample_range, PolypTestBed, "Polyp", grad_fn)
       # collect_gradient_data(sample_range, ImagenetteTestBed, "imagenette", grad_fn)
       # collect_gradient_data(sample_range, NjordTestBed, "Njord", grad_fn)


if __name__ == '__main__':
    from features import *
    torch.multiprocessing.set_start_method('spawn')
    grad_data(5)
    # sample_range = [50, 100, 200, 500]
    # sample_range = [100]
    # collect_severitydata()
    # collect_all_data(sample_range)
#     bench = NjordTestBed(10)
    # bench.split_datasets()