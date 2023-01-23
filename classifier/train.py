from pytorch_lightning import Trainer
from torchvision.models import resnet34
from domain_datasets import build_nico_dataset
from utils import wrap_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from classifier.resnetclassifier import ResNetClassifier
from torchvision.datasets import MNIST, CIFAR10,CIFAR100


# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.

def train_classifier():
    import os
    # model = ResNetClassifier(len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/autumn")), 34, transfer=False)
    # model =  ResNetClassifier(10, 34, transfer=False, batch_size=100)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)

    # model = ResNetClassifier.load_from_checkpoint("CIFAR10_logs/lightning_logs/version_2/checkpoints/epoch=20-step=1050000.ckpt", num_classes=10,
    #                       resnet_version=34).to("cuda")
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((64,64)),
                        transforms.ToTensor(), ])

    # train_set, val_set = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
    train_set = CIFAR10("../../Datasets/cifar10", train=True, transform=trans)
    val_set = CIFAR10("../../Datasets/cifar10", train=False, transform=trans)
    # train_set = MNIST("../../Datasets/mnist", train=True, transform=trans)
    # val_set = MNIST("../../Datasets/mnist", train=False, transform=trans)
    tb_logger = TensorBoardLogger(save_dir=f"{type(train_set).__name__}_logs")
    train_set = wrap_dataset(train_set)
    val_set = wrap_dataset(val_set)
    trainer = Trainer(gpus=[0], max_epochs=200, logger=tb_logger)

    # trainer.fit(model, train_dataloaders=DataLoader(train_set, shuffle=True, num_workers=10),
    #             val_dataloaders=DataLoader(val_set, shuffle=True, num_workers=10))

if __name__ == '__main__':
    train_classifier()