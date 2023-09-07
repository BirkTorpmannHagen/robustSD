import os
from torch.utils.data import ConcatDataset
import warnings
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
from domain_datasets import CIFAR10wNoise
import torch.utils.data as data
from domain_datasets import *
from torch.utils.data import RandomSampler
from classifier.resnetclassifier import ResNetClassifier
from ooddetectors import *
from torch.nn import functional as F
from classifier.cifarresnet import get_cifar, cifar10_pretrained_weight_urls
class BaseTestBed:
    def __init__(self, sample_size):
        self.sample_size = sample_size


    def compute_losses(self, loaders):
        pass

    def ind_loader(self):
        pass

    def ood_loaders(self):
        pass

    def ind_val_loaders(self):
        pass

class NicoTestBed(BaseTestBed):

    def __init__(self, sample_size):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
                                                 transforms.Resize((512, 512)),
                                                 transforms.ToTensor(), ])
        self.num_classes = num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))

        self.ind, self.ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, self.trans, self.trans, context="dim", seed=0)
        self.num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
        self.contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
        try:
            # self.classifier = ResNetClassifier.load_from_checkpoint(
            #     "lightning_logs/version_0/checkpoints/epoch=199-step=1998200.ckpt", num_classes=num_classes,
            #     resnet_version=34).to("cuda").eval()
            self.classifier = ResNetClassifier.load_from_checkpoint(
                "NICODataset_logs/lightning_logs/version_10/checkpoints/epoch=417-step=4176238.ckpt", num_classes=num_classes,
                resnet_version=18).to("cuda").eval()

        except FileNotFoundError:
            warnings.warn("No classifier found, using f:x->0")
            self.classifier = lambda x: torch.zeros(x.shape[0], num_classes).to("cuda")
        self.rep_model = self.classifier
        # self.rep_model = ResNetVAE().to("cuda").eval()

        # self.vae_experiment = VAEXperiment(self.rep_model, yaml.safe_load(open("vae/configs/vae.yaml")))
        # self.vae_experiment.load_state_dict(torch.load("vae_logs/nico_dim/version_7/checkpoints/last.ckpt")["state_dict"])


    def compute_losses(self, loader):
        losses = torch.zeros(len(loader)).to("cuda")
        print("computing losses")
        for i, (x, y, _) in tqdm(enumerate(loader), total=len(loader)):
            x = x.to("cuda")
            y = y.to("cuda")
            yhat = self.classifier(x)
            losses[i]=F.cross_entropy(yhat, y).item()
        return losses.cpu().numpy()

    def ind_loader(self):
        return DataLoader(self.ind, shuffle=False, num_workers=20)


    def ood_loaders(self):
        test_datasets = [build_nico_dataset(1, "../../Datasets/NICO++", 0.2, self.trans, self.trans, context=context, seed=0)[1] for context in self.contexts if context!="dim"]

        oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=self.num_classes), num_workers=20),
                     DataLoader(test_dataset,
                                sampler=ClusterSampler(test_dataset, self.rep_model, sample_size=self.sample_size), num_workers=20),
                     DataLoader(test_dataset, sampler=RandomSampler(test_dataset), num_workers=20)] for test_dataset in test_datasets]

        dicted = [dict([(sampler, loader) for sampler, loader in zip(["ClassOrderSampler", "ClusterSampler", "RandomSampler"], ood)]) for ood in oods]
        double_dicted = dict([(context, dicted) for context, dicted in zip(self.contexts, dicted)])
        return double_dicted

    def ind_val_loaders(self):
        loaders =  {"ind": dict([(sampler.__class__.__name__, DataLoader(self.ind_val, sampler=sampler, num_workers=20)) for sampler in [ClassOrderSampler(self.ind_val, num_classes=self.num_classes),
                                                                              ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
                                                                              RandomSampler(self.ind_val)]])}
        return loaders

    

class CIFAR10TestBed(BaseTestBed):
    def __init__(self, sample_size):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(), ])

        # classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to(
        #     "cuda").eval()
        # torch.save(classifier, "cifar10_model.pt")
        # for module in classifier.modules():
        #     print(module)
        self.classifier = get_cifar("resnet32", layers= [5]*3, model_urls=cifar10_pretrained_weight_urls,progress=True, pretrained=True).cuda().eval()

        self.rep_model = WrappedCIFAR10Resnet(self.classifier)
        # config = yaml.safe_load(open("vae/configs/vae.yaml"))
        # self.rep_model = VanillaVAE(3,512).to("cuda").eval()
        # self.rep_model = ResNetVAE().cuda().eval()
        # vae_exp = VAEXperiment(self.rep_model, config)
        # vae_exp.load_state_dict(
        #     torch.load("vae_logs/nico_dim/version_4/checkpoints/epoch=2-step=37500.ckpt")[
        #         "state_dict"])
        self.num_classes = 10
        self.ind_val = CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans)


    def ind_loader(self):
        return DataLoader(
            CIFAR10wNoise("../../Datasets/cifar10", train=True, transform=self.trans), shuffle=False, num_workers=20)

    def ind_val_loaders(self):
        loaders = {"ind": dict(
            [(sampler.__class__.__name__, DataLoader(self.ind_val, sampler=sampler, num_workers=20)) for sampler in
             [ClassOrderSampler(self.ind_val, num_classes=self.num_classes),
              ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
              RandomSampler(self.ind_val)]])}
        return loaders

    def ood_loaders(self):
        ood_sets = [CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans, noise_level=noise_val)
                                       for noise_val in np.linspace(0.1, 0.50, 5)]
        # self.oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=10)),
        #          DataLoader(test_dataset, sampler=ClusterSampler(test_dataset, self.classifier, sample_size=self.sample_size)),
        #          DataLoader(test_dataset, sampler=RandomSampler(test_dataset))] for test_dataset in ood_sets]
        oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=10)),
                 DataLoader(test_dataset, sampler=ClusterSampler(test_dataset, self.rep_model, sample_size=self.sample_size), num_workers=20),
                 DataLoader(test_dataset, sampler=RandomSampler(test_dataset), num_workers=20)] for test_dataset in ood_sets]
        dicted = [dict([(sampler, loader) for sampler, loader in zip(["ClassOrderSampler", "ClusterSampler", "RandomSampler"], ood)]) for ood in oods]
        double_dicted = dict(zip(["noise_{}".format(noise_val) for noise_val in np.linspace(0.1, 0.20, 3)], dicted))
        return double_dicted
        # return 0 #DEBUG
    def compute_losses(self, loader):
        losses = torch.zeros(len(loader) ).to("cuda")
        print("computing losses")
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            x = data[0].to("cuda")
            y = data[1].to("cuda")
            yhat = self.classifier(x)
            losses[i]=F.cross_entropy(yhat, y).item()
        return losses.cpu().numpy()