from pytorch_lightning import Trainer
from torchvision.models import resnet34
from domain_datasets import build_polyp_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from segmentation_models_pytorch import DeepLabV3Plus
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.losses import JaccardLoss
from segmentation_models_pytorch.metrics import get_stats, iou_score
import sys
import warnings
from segmentation_models_pytorch.base import SegmentationHead

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


class SegmentationModel(pl.LightningModule):
    def __init__(self, resnet_version, transfer=True,
                 optimizer='adam', lr=1e-3, batch_size=16):
        super().__init__()

        self.__dict__.update(locals())
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        self.criterion = JaccardLoss(mode="binary")
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        # Using a pretrained ResNet backbone
        self.segmentor = DeepLabV3Plus()
        self.encoder = self.segmentor.encoder
        # replace final layer for fine tuning
        self.decoder = self.segmentor.decoder
        self.latent_dim = self.get_encoding_size()



    def forward(self, X):
        return self.segmentor(X)

    def get_encoding_size(self):
        dummy = torch.zeros((1,3,512,512))
        return self.segmentor.encoder(dummy)[-2].flatten(1).shape[-1]
    def encode(self, X):
        code =  self.segmentor.encoder(X)[-2].flatten(1).squeeze(-1)
        return code

    def compute_loss(self, x, y):
        out = self.segmentor(x)
        return self.criterion(out, y)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y, context = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = torch.mean(iou_score(*get_stats(preds,y, mode="binary", threshold=0.5)))
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_iou", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y, context = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = torch.mean(iou_score(*get_stats(preds,y, mode="binary", threshold=0.5)))
        # perform logging
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y, context = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = torch.mean(iou_score(*get_stats(preds,y, mode="binary", threshold=0.5)))
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_iou", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
