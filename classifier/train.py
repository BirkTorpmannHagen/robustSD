from pytorch_lightning import Trainer
from torchvision.models import resnet34
from domain_datasets import build_nico_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser

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


# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.

def train_classifier():
    import os
    model = ResNetClassifier(len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/autumn")), 34)
    trainer = Trainer(gpus=[0], max_epochs=200)
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    train_set, val_set = build_nico_dataset(1, "../../Datasets/NICO++", 0.1, trans, trans, context="dim", seed=0)
    trainer.fit(model, train_dataloaders=DataLoader(train_set, shuffle=True, num_workers=4),val_dataloaders=DataLoader(val_set, shuffle=True, num_workers=4))

if __name__ == '__main__':
    train_classifier()