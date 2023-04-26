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

def create_testbed(name, sample_size):
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    if name=="nico":
        ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
        num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
        classifier = ResNetClassifier.load_from_checkpoint(
            "lightning_logs/version_0/checkpoints/epoch=199-step=1998200.ckpt", num_classes=num_classes,
            resnet_version=34).to("cuda").eval()
        rep_model = classifier
        ood_sets = [build_nico_dataset(build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context=i, seed=0)) for i in os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")]
        oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=num_classes)),
                 DataLoader(test_dataset, sampler=ClusterSampler(test_dataset, classifier, sample_size=sample_size)),
                 DataLoader(test_dataset, sampler=RandomSampler(test_dataset))] for test_dataset in ood_sets]

    elif name=="polyp":
        cvc_train_set, cvc_val_set = build_polyp_dataset("../../Datasets/Polyps/CVC-ClinicDB", "CVC", 0)
        kvasir_train_set, kvasir_val_set = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
        etis_train, etis_val = build_polyp_dataset("../../Datasets/Polyps/ETIS-LaribPolypDB", "Etis", 0)
        ind = ConcatDataset((cvc_train_set, kvasir_train_set, kvasir_val_set))

        ind_val = cvc_val_set
        ood_dataset = ConcatDataset([etis_train, etis_val])
        config = yaml.safe_load(open("vae/configs/vae.yaml"))
        rep_model = ResNetVAE().to("cuda").eval()
        vae_exp = VAEXperiment(rep_model, config)
        vae_exp.load_state_dict(
            torch.load("vae_logs/nico_dim/version_41/checkpoints/last.ckpt")[
                "state_dict"])
        classifier = SegmentationModel.load_from_checkpoint(
            "segmentation_logs/lightning_logs/version_11/checkpoints/epoch=142-step=23023.ckpt").to("cuda")
        classifier.eval()
        oods = [DataLoader(ood_dataset, sampler=i) for i in [ClusterSampler(ood_dataset, classifier, sample_size=sample_size), RandomSampler(ind_val),
         SequentialSampler(ood_dataset)]]

    elif name=="njord":
        trans = transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.ToTensor(), ])
        config = yaml.safe_load(open("vae/configs/vae.yaml"))
        rep_model = VanillaVAE(3, 128).to("cuda").eval()
        ind, ind_val, ood = build_njord_dataset()
        vae_exp = VAEXperiment(rep_model, config)
        vae_exp.load_state_dict(
            torch.load("njord_vae/VanillaVAE/version_0/checkpoints/epoch=16-step=31109.ckpt")[
                "state_dict"])
        classifier = lambda x: 0
        oods = [DataLoader(ood, sampler=i) for i in [RandomSampler(ood), ClusterSampler(ood, rep_model, sample_size),
                                     SequentialSampler(ood)]]
    elif name=="cifar":
        trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(), ])
        ind = wrap_dataset(CIFAR10("../../Datasets/cifar10", train=True, transform=trans))

        ind_val = wrap_dataset(CIFAR10("../../Datasets/cifar10", train=False, transform=trans))

        classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to(
            "cuda").eval()
        classifier = wrap_model(classifier)
        rep_model = classifier
        ood_sets = [transform_dataset(ind_val,lambda x: x + torch.randn_like(x) * noise_val) for noise_val in np.linspace(0, 0.20, 11)]
        oods =[[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=10)),
                 DataLoader(test_dataset, sampler=ClusterSampler(test_dataset, classifier, sample_size=sample_size)),
                 DataLoader(test_dataset, sampler=RandomSampler(test_dataset))] for test_dataset in ood_sets]
    elif name=="mnist":
        trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(), ])
        ind = wrap_dataset(MNIST("../../Datasets/mnist", train=True, transform=trans))
        ind_val = wrap_dataset(CIFAR10("../../Datasets/cifar10", train=False, transform=trans))
        classifier = ResNetClassifier.load_from_checkpoint(
            "MNIST_logs/lightning_logs/version_0/checkpoints/epoch=40-step=2460000.ckpt", num_classes=10)
        rep_model = classifier
        ood_sets = [transform_dataset(ind_val,lambda x: x + torch.randn_like(x) * noise_val) for noise_val in np.linspace(0, 0.20, 11)]
        oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=10)),
                 DataLoader(test_dataset, sampler=ClusterSampler(test_dataset, classifier, sample_size=sample_size)),
                 DataLoader(test_dataset, sampler=RandomSampler(test_dataset))] for test_dataset in ood_sets]
    else:
        raise NotImplementedError
    return {"classifier": classifier, "rep_model": rep_model, "ind": ind, "ind_val": ind_val, "oods": oods}

class BaseTestBed:
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def compute_losses(self):
        pass

    def compute_distshift(self):
        pass

class NicoTestBed(BaseTestBed):

    def __init__(self, sample_size):
        super().__init__(sample_size)
        ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
        self.num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
        classifier = ResNetClassifier.load_from_checkpoint(
            "lightning_logs/version_0/checkpoints/epoch=199-step=1998200.ckpt", num_classes=num_classes,
            resnet_version=34).to("cuda").eval()
        rep_model = classifier
        ood_sets = [build_nico_dataset(build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context=i, seed=0)) for i in os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")]
        oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=num_classes)),
                 DataLoader(test_dataset, sampler=ClusterSampler(test_dataset, classifier, sample_size=sample_size)),
                 DataLoader(test_dataset, sampler=RandomSampler(test_dataset))] for test_dataset in ood_sets]
    




