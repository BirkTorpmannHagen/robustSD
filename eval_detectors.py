import os
from torch.utils.data import ConcatDataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def transform_dataset(dataset, transform):
    class NewDataset(data.Dataset):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

        def __getitem__(self, index):
            image, label, context = self.dataset[index]
            image = transform(image)
            return image, label, context

        def __len__(self):
            return len(self.dataset)

    return NewDataset(dataset)





def collect_data(name):

    for sample_size in [10, 20, 50, 100, 200, 500]:
        exp_dict = create_testbed(name, sample_size=sample_size)
        classifier = exp_dict["classifier"]
        rep_model = exp_dict["rep_model"]
        ind = exp_dict["ind"]
        ind_val = exp_dict["ind_val"]
        oods = exp_dict["oods"]
        aconfig = {"device":"cuda"}
        merged = []
        ds = robustSD(rep_model, classifier, aconfig)

        for i, test_dataset in enumerate(oods):
            for sampled_loader in test_dataset:
                data = ds.compute_pvals_and_loss(ind, test_dataset, ood_sampler=sampler,
                                                 sample_size=sample_size, ind_dataset_name=f"{name}_ind",
                                                 ood_dataset_name=f"{name}_{i}", k=k)
                data["sampler"] = type(sampled_loader).__name__
                merged.append(data)


def eval_nico():
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))

    ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
    # ind, ind_val = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    config = yaml.safe_load(open("vae/configs/vae.yaml"))
    # model = ResNetVAE().to("cuda").eval()
    # vae_exp = VAEXperiment(model, config)
    # vae_exp.load_state_dict(
    #     torch.load("/home/birk/Projects/robustSD/vae_logs/nico_dim/version_40/checkpoints/epoch=8-step=22482.ckpt")[
    #         "state_dict"])
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = ResNetClassifier.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=199-step=1998200.ckpt", num_classes=num_classes, resnet_version=34).to("cuda")
    # classifier = SegmentationnicoModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_1/checkpoints/epoch=48-step=2450.ckpt").to("cuda").eval()

    aconfig = {"device":"cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    # ds = robustSD(model, classifier, aconfig)

    # print("ood")
    ind_dataset_name = "nico"
    columns=["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
    merged = []
    k=5
    try:
        for context in os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train"):
            test_dataset = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context=context, seed=0)[1]
            for sample_size in [10, 20, 50, 100, 200, 500]:
                for sampler in [ClassOrderSampler(test_dataset, num_classes=num_classes), ClusterSampler(test_dataset, classifier, sample_size=sample_size), RandomSampler(test_dataset)]:
                    data = ds.compute_pvals_and_loss(ind, test_dataset, ood_sampler=sampler,
                                                     sample_size=sample_size, ind_dataset_name="nico_dim", ood_dataset_name=f"nico_{context}",k=k)
                    data["sampler"]=type(sampler).__name__
                    merged.append(data)
    except KeyboardInterrupt:
        final = pd.concat(merged)
        final.to_csv(f"{type(ds.rep_model).__name__}_dim_k{k}.csv")
    final = pd.concat(merged)
    final.to_csv(f"nico_{type(ds.rep_model).__name__}_k{k}.csv")
    print(final.head(10))


def eval_nico_for_k():
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))

    ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
    # ind, ind_val = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    config = yaml.safe_load(open("vae/configs/vae.yaml"))

    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = ResNetClassifier.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=199-step=1998200.ckpt", num_classes=num_classes, resnet_version=34).to("cuda")
    # classifier = SegmentationnicoModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_1/checkpoints/epoch=48-step=2450.ckpt").to("cuda").eval()

    aconfig = {"device":"cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    # print("ood")
    ind_dataset_name = "nico"
    columns=["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
    merged = []
    k=2
    try:
        for context in os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train"):
            test_dataset = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context=context, seed=0)[1]
            for sample_size in [10, 20, 50, 100, 200, 500, 1000, 10000]:
                for k in [1, 5, 10, 15, 20, 25]:
                    data = ds.compute_pvals_and_loss(ind, test_dataset, ood_sampler=ClusterSampler(test_dataset),
                                                     sample_size=sample_size, ind_dataset_name="nico_dim", ood_dataset_name=f"nico_{context}",k=k)
                    data["k"]=k
                    merged.append(data)
    except KeyboardInterrupt:
        final = pd.concat(merged)
        final.to_csv(f"{type(ds.rep_model).__name__}_dim_all_ks_ClusterSampler-incomplete.csv")
    final = pd.concat(merged)
    final.to_csv(f"{type(ds.rep_model).__name__}_dim_all_ks_ClusterSampler.csv")
    print(final.head(10))
def nico_correlation():
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))

    ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
    # ind, ind_val = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    # config = yaml.safe_load(open("vae/configs/vae.yaml"))
    # model = ResNetVAE().to("cuda").eval()
    # vae_exp = VAEXperiment(model, config)
    # vae_exp.load_state_dict(
    #     torch.load("vae_logs/nico_dim/version_40/checkpoints/last.ckpt")[
    #         "state_dict"])
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = ResNetClassifier.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=199-step=1998200.ckpt", num_classes=num_classes, resnet_version=34).to("cuda")
    # classifier = SegmentationModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_1/checkpoints/epoch=48-step=2450.ckpt").to("cuda").eval()

    aconfig = {"device":"cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    # ds = robustSD(model, classifier, aconfig)

    # print("ood")
    merged = []
    for noise_val in np.linspace(0, 0.20, 11):
        for sample_size in [10, 20, 50, 100]:
            ood_dataset = transform_dataset(ind_val, lambda x: x + torch.randn_like(x) * noise_val)
            for sampler_type in [ClusterSampler(ood_dataset, classifier, sample_size=sample_size), RandomSampler(ood_dataset),
                                 ClassOrderSampler(ood_dataset, num_classes=num_classes)]:
                data = ds.compute_pvals_and_loss(ind, ind_val,
                                         ood_sampler=sampler_type,
                                         sample_size=sample_size, ind_dataset_name="nico_dim",
                                         ood_dataset_name=f"nico_{noise_val}", k=5)
                data["sampler"] = type(sampler_type).__name__
                merged.append(data)
                print(data.head(10))
    final = pd.concat(merged)
    final.to_csv(f"lp_data_nico_noise.csv")
    print(final.head(10))

def eval_polyp():
    trans = transforms.Compose([transforms.Resize((512, 512)),
                                transforms.ToTensor()])
    cvc_train_set, cvc_val_set = build_polyp_dataset("../../Datasets/Polyps/CVC-ClinicDB", "CVC", 0)
    kvasir_train_set, kvasir_val_set = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    etis_train, etis_val = build_polyp_dataset("../../Datasets/Polyps/ETIS-LaribPolypDB", "Etis", 0)
    ind = ConcatDataset((cvc_train_set, kvasir_train_set, kvasir_val_set))

    ind_val = cvc_val_set
    ood = ConcatDataset([etis_train, etis_val])

    config = yaml.safe_load(open("vae/configs/vae.yaml"))
    model = ResNetVAE().to("cuda").eval()
    vae_exp = VAEXperiment(model, config)
    vae_exp.load_state_dict(
        torch.load("vae_logs/nico_dim/version_41/checkpoints/last.ckpt")[
            "state_dict"])
    classifier = SegmentationModel.load_from_checkpoint(
        "segmentation_logs/lightning_logs/version_11/checkpoints/epoch=142-step=23023.ckpt").to("cuda")
    classifier.eval()
    aconfig = {"device": "cuda"}
    ds = robustSD(model, classifier, aconfig)

    merged = []
    for name, dataset in zip(["ind", "ood"], [ind_val, ood]):
        for sample_size in [10, 20, 50, 100, 200, 500, 1000]:
            for sampler_type in [ClusterSampler(dataset, classifier, sample_size=sample_size), RandomSampler(ind_val), SequentialSampler(dataset)]:
                data = ds.compute_pvals_and_loss(ind, dataset,
                                         ood_sampler=sampler_type,
                                         sample_size=sample_size, ind_dataset_name="polyp_ind",
                                         ood_dataset_name=f"polyp_{name}", k=5)
                data["sampler"]=type(sampler_type).__name__
                merged.append(data)
                print(data.head(10))
    final = pd.concat(merged)
    final.to_csv(f"lp_data_polyps.csv")
    print(final.head(10))

def eval_cifar10():
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(), ])
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))
    test = CIFAR10("../../Datasets/cifar10", train=True, transform=trans)
    ind = wrap_dataset(CIFAR10("../../Datasets/cifar10", train=True, transform=trans))

    ind_val = wrap_dataset(CIFAR10("../../Datasets/cifar10", train=False, transform=trans))

    classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to("cuda").eval()
    classifier = wrap_model(classifier)
    aconfig = {"device": "cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    ind_dataset_name = "cifar10"
    columns = ["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]

    merged = []
    for noise_val in np.linspace(0, 0.20, 11):
        for sample_size in [10, 20, 50, 100, 200, 500, 1000]:
            ood_set = transform_dataset(ind_val,lambda x: x + torch.randn_like(x) * noise_val )
            for sampler_type in [ClusterSampler(ood_set, classifier, sample_size=sample_size), RandomSampler(ood_set), ClassOrderSampler(ood_set, num_classes=10)]:
                data = ds.compute_pvals_and_loss(ind, ood_set,
                                         ood_sampler=sampler_type,
                                         sample_size=sample_size, ind_dataset_name="cifar10",
                                         ood_dataset_name=f"cifar10_{noise_val}", k=5)
                data["sampler"]=type(sampler_type).__name__
                merged.append(data)
                print(data.head(10))
    final = pd.concat(merged)
    final.to_csv(f"lp_data_cifar10_noise.csv")
    print(final.head(10))

def eval_mnist():
    trans = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor(), ])
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))
    ind = wrap_dataset(MNIST("../../Datasets/mnist", train=True, transform=trans))
    ind_val = wrap_dataset(MNIST("../../Datasets/mnist", train=False, transform=trans))

    classifier = ResNetClassifier.load_from_checkpoint("MNIST_logs/lightning_logs/version_0/checkpoints/epoch=40-step=2460000.ckpt",
                                                       num_classes=10, resnet_version=34,transfer=True, batch_size=100).to("cuda")
    classifier.eval()

    aconfig = {"device": "cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    # ds = robustSD(model, classifier, aconfig)

    # print("ood")
    merged = []
    for noise_val in np.linspace(0, 0.20, 11):
        for sample_size in [10, 100]:
            ood_set = transform_dataset(ind_val, lambda x: x + torch.randn_like(x) * noise_val)
            for sampler_type in [ClusterSampler(ood_set, classifier, sample_size=sample_size), RandomSampler(ood_set),
                                 ClassOrderSampler(ood_set, num_classes=10)]:
                data = ds.compute_pvals_and_loss(ind, ood_set,
                                                 ood_sampler=sampler_type,
                                                 sample_size=sample_size, ind_dataset_name="mnist",
                                                 ood_dataset_name=f"mnist_{noise_val}", k=5)
                data["sampler"] = type(sampler_type).__name__
                merged.append(data)
                print(data.head(10))
    final = pd.concat(merged)
    final.to_csv(f"lp_data_mnist_noise.csv")
    print(final.head(10))
def eval_cifar10_bias_severity():
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(), ])
    contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))
    test = CIFAR10("../../Datasets/cifar10", train=True, transform=trans)
    ind = wrap_dataset(CIFAR10("../../Datasets/cifar10", train=True, transform=trans))

    ind_val = wrap_dataset(CIFAR10("../../Datasets/cifar10", train=False, transform=trans))

    # ind, ind_val = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    # config = yaml.safe_load(open("vae/configs/vae.yaml"))
    # model = ResNetVAE().to("cuda").eval()
    # vae_exp = VAEXperiment(model, config)
    # vae_exp.load_state_dict(
    #     torch.load("vae_logs/nico_dim/version_40/checkpoints/last.ckpt")[
    #         "state_dict"])
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to("cuda").eval()
    classifier = wrap_model(classifier)
    print(classifier.get_encoding_size())
    # classifier = SegmentationModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_1/checkpoints/epoch=48-step=2450.ckpt").to("cuda").eval()

    aconfig = {"device": "cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    # ds = robustSD(model, classifier, aconfig)

    # print("ood")
    ind_dataset_name = "nico"
    columns = ["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
    merged = []
    for noise_val in np.linspace(0, 0.20, 11):
        for severity in np.linspace(0.1, 1, 10):
            ood_set = transform_dataset(ind_val,lambda x: x + torch.randn_like(x) * noise_val)
            for sample_size in [10, 20, 50, 100]:
                sampler_type = ClusterSamplerWithSeverity(ood_set, classifier, sample_size=sample_size, bias_severity=severity)
                data = ds.compute_pvals_and_loss(ind, ood_set,
                                         ood_sampler=sampler_type,
                                         sample_size=sample_size, ind_dataset_name="cifar10",
                                         ood_dataset_name=f"cifar10_{noise_val}", k=5)
                data["sampler"]=f"{type(sampler_type).__name__}_{severity}"
                merged.append(data)
                print(data.head(10))
    final = pd.concat(merged)
    final.to_csv(f"lp_data_cifar10_noise_severity.csv")
    print(final.head(10))

def eval_cifar10_fork():
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(), ])
    contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))
    test = CIFAR10("../../Datasets/cifar10", train=True, transform=trans)
    ind = wrap_dataset(CIFAR10("../../Datasets/cifar10", train=True, transform=trans))

    ind_val = wrap_dataset(CIFAR10("../../Datasets/cifar10", train=False, transform=trans))

    # ind, ind_val = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    # config = yaml.safe_load(open("vae/configs/vae.yaml"))
    # model = ResNetVAE().to("cuda").eval()
    # vae_exp = VAEXperiment(model, config)
    # vae_exp.load_state_dict(
    #     torch.load("vae_logs/nico_dim/version_40/checkpoints/last.ckpt")[
    #         "state_dict"])
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to("cuda").eval()
    classifier = wrap_model(classifier)
    print(classifier.get_encoding_size())
    # classifier = SegmentationModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_1/checkpoints/epoch=48-step=2450.ckpt").to("cuda").eval()

    aconfig = {"device": "cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    # ds = robustSD(model, classifier, aconfig)

    # print("ood")
    ind_dataset_name = "nico"
    columns = ["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
    merged = []
    for noise_val in np.linspace(0, 0.20, 11):
        for sample_size in [10, 20, 50]:
            for k in [1, 5, 10, 20, 50, 100, 200]:
                data = ds.eval_synthetic(ind, ind_val, lambda x: x + torch.randn_like(x) * noise_val,
                                         sampler=ClusterSampler(ind_val, classifier, sample_size=sample_size),
                                         sample_size=sample_size, ind_dataset_name="cifar10",
                                         ood_dataset_name=f"cifar10_{noise_val}", k=k)
                data["k"]=k
                merged.append(data)
    final = pd.concat(merged)
    final.to_csv(f"lp_data_cifar10_ks.csv")
    print(final.head(10))

def eval_cifar100():
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(), ])
    contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))
    ind = wrap_dataset(CIFAR100("../../Datasets/cifar100", train=True, transform=trans))

    ind_val = wrap_dataset(CIFAR100("../../Datasets/cifar100", train=False, transform=trans))

    # ind, ind_val = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    # config = yaml.safe_load(open("vae/configs/vae.yaml"))
    # model = ResNetVAE().to("cuda").eval()
    # vae_exp = VAEXperiment(model, config)
    # vae_exp.load_state_dict(
    #     torch.load("vae_logs/nico_dim/version_40/checkpoints/last.ckpt")[
    #         "state_dict"])
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=True).to("cuda").eval()
    classifier = wrap_model(classifier)
    print(classifier.get_encoding_size())
    # classifier = SegmentationModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_1/checkpoints/epoch=48-step=2450.ckpt").to("cuda").eval()

    aconfig = {"device": "cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    # ds = robustSD(model, classifier, aconfig)

    # print("ood")
    columns = ["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
    merged = []
    for noise_val in np.linspace(0, 0.20, 11):
        for sample_size in [100]:
            data = ds.eval_synthetic(ind, ind_val, lambda x: x + torch.randn_like(x) * noise_val,
                                     sampler=ClusterSampler(ind_val, classifier, sample_size=sample_size),
                                     sample_size=sample_size, ind_dataset_name="cifar100",
                                     ood_dataset_name=f"cifar100_{noise_val}")
            merged.append(data)
    final = pd.concat(merged)
    final.to_csv(f"lp_data_cifar100_noise_clusterbias.csv")
    print(final.head(10))


def eval_njord():
    trans = transforms.Compose([transforms.Resize((512, 512)),
                                transforms.ToTensor(), ])
    config = yaml.safe_load(open("vae/configs/vae.yaml"))
    model = VanillaVAE(3, 128).to("cuda").eval()
    ind, ind_val, ood = build_njord_dataset()
    vae_exp = VAEXperiment(model, config)
    vae_exp.load_state_dict(
        torch.load("njord_vae/VanillaVAE/version_0/checkpoints/epoch=16-step=31109.ckpt")[
            "state_dict"])

    aconfig = {"device": "cuda"}
    ds = robustSD(model, None, aconfig)
    # ds = robustSD(model, classifier, aconfig)

    # print("ood")
    columns = ["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
    merged = []
    k = 1
    try:
        for name, test_dataset in zip(["ind", "ood"], [ind_val, ood]):
            for sample_size in [10, 20, 50, 100, 200, 500]:
                for sampler_type in [RandomSampler(test_dataset), ClusterSampler(test_dataset, model, sample_size),
                                     SequentialSampler(test_dataset)]:
                    data = ds.compute_pvals(ind, test_dataset, ood_sampler=sampler_type,
                                                     sample_size=sample_size, ind_dataset_name=type(ind_val).__name__+"ind",
                                                     ood_dataset_name=f"{type(test_dataset).__name__}"+name, k=k)
                    data["sampler"] = type(sampler_type).__name__
                    merged.append(data)

    except KeyboardInterrupt:
        final = pd.concat(merged)
        final.to_csv(f"{type(ds.rep_model).__name__}_k{k}_ClusterSample.csv")
    final = pd.concat(merged)
    final.to_csv(f"{type(ind_val).__name__}_{type(ds.rep_model).__name__}_k{k}.csv")
    print(final.head(10))

if __name__ == '__main__':
    # eval_mnist()
    # cifar10_bench = CIFAR10TestBed(10)
    bench = NicoTestBed(100)
    tsd = TypicalitySD(bench.rep_model, None)
    tsd.register_testbed(bench)
    tsd.compute_pvals_and_loss(100)