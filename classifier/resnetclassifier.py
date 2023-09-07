
import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt

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
                 optimizer='adam', lr=1e-5, batch_size=16,
                 transfer=False):
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

        self.latent_dim = self.get_encoding_size(-2)




    def forward(self, X):
        return self.resnet_model(X)

    def get_encoding_size(self, depth):
        dummy = torch.zeros((1,3,512,512))
        print(list(self.resnet_model.children())[-1])
        print(
            torch.nn.Sequential(*list(self.resnet_model.children())[:-1])(dummy).shape
        )
        return torch.nn.Sequential(*list(self.resnet_model.children())[:-1])(dummy).flatten(1).shape[-1]
    def get_encoding(self, X, depth=-2):
        return torch.nn.Sequential(*list(self.resnet_model.children())[:-1])(X).flatten(1)

    def compute_loss(self, x, y):
        return self.criterion(self(x), y)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y, context = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        acc = (y == torch.argmax(preds, 1)) \
            .type(torch.FloatTensor).mean()
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, context = batch
        preds = self(x)
        loss = self.compute_loss(x,y)
        acc = (y == torch.argmax(preds, 1)) \
            .type(torch.FloatTensor).mean()
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
    def test_step(self, batch, batch_idx):
        x, y, context = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (y== torch.argmax(preds, 1)) \
            .type(torch.FloatTensor).mean()
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
