from pytorch_lightning import Trainer
from torchvision.models import resnet34
from domain_datasets import build_nico_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks import ModelCheckpoint

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
    # model = ResNetClassifier.load_from_checkpoint("MNIST_logs/lightning_logs/version_0/checkpoints/epoch=40-step=2460000.ckpt", num_classes=10, resnet_version=34)
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.AutoAugment(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    val_trans = transforms.Compose([
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])

    train_set, val_set = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, val_trans, context="dim", seed=0)
    # val_set = CIFAR10("../../Datasets/cifar10", train=False, transform=trans)
    num_classes =  train_set.num_classes
    model =  ResNetClassifier(num_classes, 152, transfer=False, batch_size=32, lr=1e-5).to("cuda")
    # train_set = MNIST("../../Datasets/mnist", train=True, download=True, transform=trans)
    # val_set = MNIST("../../Datasets/mnist", train=False, download=True, transform=trans)
    tb_logger = TensorBoardLogger(save_dir=f"{type(train_set).__name__}_logs")
    # train_set = wrap_dataset(train_set)
    # val_set = wrap_dataset(val_set)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{type(train_set).__name__}_logs/checkpoints",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = Trainer(max_epochs=500, logger=tb_logger, accelerator="gpu",callbacks=checkpoint_callback )
    trainer.fit(model, train_dataloaders=DataLoader(train_set, shuffle=True, num_workers=24),
                val_dataloaders=DataLoader(val_set, shuffle=True, num_workers=24))

if __name__ == '__main__':
    train_classifier()