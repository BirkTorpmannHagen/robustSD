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
from oodsplit import FeatureOODSplitter
def train_inferredclassifier():
    import os
    model = ResNetClassifier(len(os.listdir("datasets/NICO++/track_1/public_dg_0416/train/autumn")), 34)
    trainer = Trainer(gpus=[0], max_epochs=200)
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    train_set = build_nico_dataset(1, "datasets/NICO++", 0, trans, trans, 0)[0]
    splitter = FeatureOODSplitter(train_set, split=(80,10,10))
    train_loader = splitter.get_trainloader()
    val_loader = splitter.get_valloader()
    test_loader = splitter.get_testloader()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    train_inferredclassifier()