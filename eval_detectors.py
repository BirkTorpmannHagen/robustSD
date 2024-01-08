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


def collect_gradient_data(sample_range, testbed_constructor, dataset_name, grad_fn, mode="normal"):
    for sample_size in sample_range:
        bench = testbed_constructor(rep_model="classifier", sample_size=sample_size, mode=mode)
        tsd = GradientSD(bench.classifier, grad_magnitude)
        tsd.register_testbed(bench)
        compute_stats(*tsd.compute_pvals_and_loss(sample_size),
                      fname=f"grad_data/{dataset_name}_{mode}_{grad_fn.__name__}{sample_size}.csv")

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
            tsd = GradientSD(bench.classifier, processes=5)
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

    # collect_data(sample_range, CIFAR10TestBed, "CIFAR10")
    # collect_data(sample_range, CIFAR100TestBed, "CIFAR100")
    # collect_data(sample_range, ImagenetteTestBed, "imagenette")
    # collect_data(sample_range, PolypTestBed, "Polyp")
    #collect_data(sample_range, NicoTestBed, "NICO")
    # collect_data(sample_range, NjordTestBed, "Njord")

    # collect_data(sample_range, CIFAR10TestBed, "CIFAR10", mode="severity")
    # collect_data(sample_range, CIFAR100TestBed, "CIFAR100", mode="severity")
    # collect_data(sample_range, ImagenetteTestBed, "imagenette", mode="severity")
    # collect_data(sample_range, PolypTestBed, "Polyp", mode="severity")
    # collect_data(sample_range, NicoTestBed, "NICO", mode="severity")
    # collect_data(sample_range, NjordTestBed, "Njord", mode="severity")

def grad_data():
    sample_range = [10, 20, 50, 100]
    # for grad_fn in [condition_number]:
    #     collect_gradient_data(sample_range, CIFAR10TestBed, "CIFAR10", grad_fn)
    #     collect_gradient_data(sample_range, CIFAR100TestBed, "CIFAR100", grad_fn)
    #     collect_gradient_data(sample_range, NicoTestBed, "NICO", grad_fn)
    for grad_fn in [jacobian, condition_number]:
    #     collect_gradient_data(sample_range, PolypTestBed, "Polyp", grad_fn)
        collect_gradient_data(sample_range, ImagenetteTestBed, "imagenette", grad_fn)
       # collect_gradient_data(sample_range, NjordTestBed, "Njord", grad_fn)


if __name__ == '__main__':
    from gradientfeatures import *
    grad_data()
    # for sample_size in [50, 100, 200, 500]:
    #     testbed = NicoTestBed(100, rep_model="vae", mode="normal")
    #     dsd = TypicalitySD(testbed.vae)
    #     dsd.register_testbed(testbed)
    #     compute_stats(*dsd.compute_pvals_and_loss(sample_size),
    #                   fname=f"new_data/NICO_normal_typicality.csv")

    # for feature in [jacobian]:
    #     # testbed = CIFAR10TestBed(100, rep_model="classifier", mode="normal")
    #     testbed = NicoTestBed(100, rep_model="classifier", mode="noise")
    #     dsd = GradientSD(testbed.classifier, norm_fn=feature)
    #     dsd.register_testbed(testbed)
    #     compute_stats(*dsd.compute_pvals_and_loss(100),
    #                   fname=f"{feature.__name__}_test_Nico_noise.csv")
    #
    # for feature in [condition_number, grad_magnitude]:
    #     # testbed = CIFAR10TestBed(100, rep_model="classifier", mode="normal")
    #     testbed = NicoTestBed(100, rep_model="classifier", mode="random")
    #     dsd = GradientSD(testbed.classifier, norm_fn=feature)
    #     dsd.register_testbed(testbed)
    #     compute_stats(*dsd.compute_pvals_and_loss(100),
    #                   fname=f"{feature.__name__}_test_Nico.csv")
    #
    # for feature in [condition_number, grad_magnitude]:
    #     # testbed = CIFAR10TestBed(100, rep_model="classifier", mode="normal")
    #     testbed = NicoTestBed(100, rep_model="classifier", mode="noise")
    #     dsd = GradientSD(testbed.classifier, norm_fn=feature)
    #     dsd.register_testbed(testbed)
    #     compute_stats(*dsd.compute_pvals_and_loss(100),
    #                   fname=f"{feature.__name__}_test_Nico_noise.csv")

    # torch.multiprocessing.set_start_method('spawn')
    # sample_range = [50, 100, 200, 500]
    # sample_range = [100]
    # collect_severitydata()
# ''    collect_all_data(sample_range)
#     bench = NjordTestBed(10)
    # bench.split_datasets()