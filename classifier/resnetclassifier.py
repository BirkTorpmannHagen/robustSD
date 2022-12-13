from pytorch_lightning import Trainer
from torchvision.models import resnet34
from domain_datasets import build_dataset
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


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, resnet_version,
                 optimizer='adam', lr=1e-3, batch_size=16,
                 transfer=True, tune_fc_only=True):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        self.num_classes = num_classes
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        self.criterion = nn.CrossEntropyLoss()
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if tune_fc_only:  # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False



    def forward(self, X):
        return self.resnet_model(X)

    def compute_loss(self, x, y):
        return self.criterion(self(x), y)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        transform = transforms.Compose([
            transforms.Resize((512, 5121)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor()
        ])
        img_train= build_dataset(1, "datasets/NICO++", 0.1, transform, transform, 0)[0]
        return DataLoader(img_train, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, y, context = batch
        preds = self(x)
        y = F.one_hot(y, num_classes=self.num_classes).float()


        loss = self.criterion(preds, y)
        acc = (torch.argmax(y, 1) == torch.argmax(preds, 1)) \
            .type(torch.FloatTensor).mean()
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def val_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        img_val = build_dataset(1, "datasets/NICO++", 0.1, transform, transform, 0)[1]

        return DataLoader(img_val, batch_size=1, shuffle=False)

    def validation_step(self, batch, batch_idx):
        x, y, context = batch
        preds = self(x)
        y = F.one_hot(y, num_classes=self.num_classes).float()

        loss = self.criterion(preds, y)
        acc = (torch.argmax(y, 1) == torch.argmax(preds, 1)) \
            .type(torch.FloatTensor).mean()
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        return self.val_dataloader()

    def test_step(self, batch, batch_idx):
        x, y, context = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()

        loss = self.criterion(preds, y)
        acc = (torch.argmax(y, 1) == torch.argmax(preds, 1)) \
            .type(torch.FloatTensor).mean()
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
