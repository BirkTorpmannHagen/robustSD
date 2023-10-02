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

from njord.utils.loss import ComputeLoss
from njord.val import fetch_model
from njord.utils.dataloaders import LoadImagesAndLabels
import segmentation_models_pytorch as smp

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

class NjordTestBed(BaseTestBed):
    def __init__(self, sample_size):
        super().__init__(sample_size)
        ind, ind_val, ood = build_njord_dataset()
        self.rep_model = ResNetVAE().to("cuda").eval()
        self.classifier = fetch_model("njord/runs/train/exp4/weights/best.pt")

        self.classifier.hyp = yaml.safe_load(open("/home/birk/Projects/robustSD/njord/data/hyps/hyp.scratch-low.yaml", "r"))
        self.loss = ComputeLoss(self.classifier)
        self.vae_exp = VAEXperiment(self.rep_model, yaml.safe_load(open("vae/configs/vae.yaml")))
        self.vae_exp.load_state_dict(
            torch.load("vae_logs/Njord/version_3/checkpoints/last.ckpt")[
                "state_dict"])
        self.ind, self.ind_val, self.ood = build_njord_dataset()
        self.collate_fn = LoadImagesAndLabels.collate_fn

    def loader(self, dataset, sampler):
        return DataLoader(dataset, sampler=sampler, collate_fn=self.collate_fn, num_workers=0)

    def compute_losses(self, loader):
        losses = torch.zeros(len(loader))
        for i, (x, targets, paths, shapes) in enumerate(loader):

            x = x.half()/255
            x = x.cuda()
            targets = targets.cuda()
            preds, train_out = self.classifier(x)
            _, loss = self.loss(train_out, targets)
            losses[i]=loss.mean().item()
        return losses

    def ind_loader(self):
        return self.loader(self.ind, sampler=RandomSampler(self.ind))

    def ood_loaders(self):
        samplers = [ClusterSampler(self.ood, self.rep_model, sample_size=self.sample_size),
                                  SequentialSampler(self.ood), RandomSampler(self.ood)]
        loaders =  {"ood": dict([[sampler.__class__.__name__,  self.loader(self.ood, sampler=sampler)] for sampler in
                                 samplers])}
        return loaders


    def ind_val_loaders(self):
        samplers = [ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
                                  SequentialSampler(self.ind_val), RandomSampler(self.ind_val)]

        loaders =  {"ind": dict([ [sampler.__class__.__name__,  self.loader(self.ind_val, sampler=sampler)] for sampler in
                                 samplers])}
        return  loaders


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
        print(self.contexts)
        self.contexts.remove("dim")
            # self.classifier = ResNetClassifier.load_from_checkpoint(
            #     "lightning_logs/version_0/checkpoints/epoch=199-step=1998200.ckpt", num_classes=num_classes,
            #     resnet_version=34).to("cuda").eval()
        self.classifier = ResNetClassifier.load_from_checkpoint(
           "NICODataset_logs/checkpoints_newer/epoch=398-step=3986409.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()


        # self.rep_model = self.classifier
        # self.rep_model = ResNetVAE().to("cuda").eval()
        self.rep_model = self.classifier
        # self.vae_experiment = VAEXperiment(self.rep_model, yaml.safe_load(open("vae/configs/vae.yaml")))
        # self.vae_experiment.load_state_dict(torch.load("/home/birk/Projects/robustSD/vae_logs/nico_dim/version_7/checkpoints/last.ckpt")["state_dict"])


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
        test_datasets = [build_nico_dataset(1, "../../Datasets/NICO++", 0.2, self.trans, self.trans, context=context, seed=0)[1] for context in self.contexts]

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
        config = yaml.safe_load(open("vae/configs/vae.yaml"))
        self.vae = ResNetVAE().cuda().eval()
        vae_exp = VAEXperiment(self.vae, config)
        vae_exp.load_state_dict(
            torch.load("vae_logs/CIFAR10/version_5/checkpoints/epoch=3-step=3128.ckpt")[
                "state_dict"])
        self.num_classes = 10
        # self.ind_val = CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans, noise_level=0)
        self.ind, self.ind_val = torch.utils.data.random_split(CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans),[0.5, 0.5])


    def ind_loader(self):
        # return DataLoader(
        #     CIFAR10wNoise("../../Datasets/cifar10", train=True, transform=self.trans,noise_level=0), shuffle=False, num_workers=20)
        return DataLoader(
            self.ind, shuffle=False, num_workers=20)

    def ind_val_loaders(self):
        loaders = {"ind": dict(
            [(sampler.__class__.__name__, DataLoader(self.ind_val, sampler=sampler, num_workers=20)) for sampler in
             [ClassOrderSampler(self.ind_val, num_classes=self.num_classes),
              ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
              RandomSampler(self.ind_val)]])}
        return loaders

    def ood_loaders(self):
        ood_sets = [CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans, noise_level=noise_val)
                                       for noise_val in np.arange(0.1, 0.3, 0.05)]
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

class PolypTestBed(BaseTestBed):
    def __init__(self, sample_size):
        super().__init__(sample_size)
        self.ind, self.val, self.ood = build_polyp_dataset("../../Datasets/Polyp")
        self.rep_model = smp.DeepLabV3Plus()
        dict = torch.load("dict")
        self.rep_model.load_state_dict(dict)
        trans = transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.ToTensor()])
        cvc_train_set, cvc_val_set = build_polyp_dataset("../../Datasets/Polyps/CVC-ClinicDB", "CVC", 0)
        kvasir_train_set, kvasir_val_set = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
        etis_train, etis_val = build_polyp_dataset("../../Datasets/Polyps/ETIS-LaribPolypDB", "Etis", 0)
        self.ind = ConcatDataset((cvc_train_set, kvasir_train_set, kvasir_val_set))
        self.ind_val = cvc_val_set
        self.ood = ConcatDataset([etis_train, etis_val])

        config = yaml.safe_load(open("vae/configs/vae.yaml"))
        model = ResNetVAE().to("cuda").eval()
        vae_exp = VAEXperiment(model, config)
        vae_exp.load_state_dict(
            torch.load("vae_logs/nico_dim/version_41/checkpoints/last.ckpt")[
                "state_dict"])
        classifier = SegmentationModel.load_from_checkpoint(
            "segmentation_logs/lightning_logs/version_11/checkpoints/epoch=142-step=23023.ckpt").to("cuda")
        classifier.eval()
        self.rep_model = classifier

    def ind_loader(self):
        return self.loader(self.ind, sampler=RandomSampler(self.ind))

    def ood_loaders(self):
        samplers = [ClusterSampler(self.ood, self.rep_model, sample_size=self.sample_size),
                                  SequentialSampler(self.ood), RandomSampler(self.ood)]
        loaders =  {"ood": dict([[sampler.__class__.__name__,  self.loader(self.ood, sampler=sampler)] for sampler in
                                 samplers])}
        return loaders


    def ind_val_loaders(self):
        samplers = [ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
                                  SequentialSampler(self.ind_val), RandomSampler(self.ind_val)]

        loaders =  {"ind": dict([ [sampler.__class__.__name__,  self.loader(self.ind_val, sampler=sampler)] for sampler in
                                 samplers])}
        return loaders
