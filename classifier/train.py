from pytorch_lightning import Trainer
from torchvision.models import resnet34
from domain_datasets import build_nico_dataset, build_imagenette_dataset
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

def train_classifier(train_set, val_set):
    num_classes =  train_set.num_classes
    model =  ResNetClassifier(num_classes, 101, transfer=False, batch_size=32, lr=1e-3).to("cuda")
    model = ResNetClassifier.load_from_checkpoint("Imagenette_logs/checkpoints/epoch=82-step=24568.ckpt", num_classes=num_classes, resnet_version=101, batch_size=16, lr=1e-7).cuda().eval()
    tb_logger = TensorBoardLogger(save_dir=f"{type(train_set).__name__}_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{type(train_set).__name__}_logs/checkpoints",
        save_top_k=3,
        verbose=True,
        monitor="val_acc",
        mode="max"
    )

    model.eval()
    # ResNetClassifier.load_from_checkpoint("Imagenette_logs/checkpoints/epoch=82-step=24568.ckpt", resnet_version=101, nj
    trainer = Trainer(max_epochs=1000, logger=tb_logger, accelerator="gpu",callbacks=checkpoint_callback)
    trainer.validate(model, DataLoader(val_set, shuffle=False, batch_size=1, num_workers=1))

    # trainer.fit(model, train_dataloaders=DataLoader(train_set, shuffle=True, batch_size=32, num_workers=24),
    #             val_dataloaders=DataLoader(val_set, shuffle=True, batch_size=32, num_workers=24))

if __name__ == '__main__':
    #NICO
    size = 256
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((size,size)),
                        transforms.ToTensor(), ])
    val_trans = transforms.Compose([
                        transforms.Resize((size,size)),
                        transforms.ToTensor(), ])

    # train_set, val_set = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, val_trans, context="dim", seed=0)
    train_set, val_set = build_imagenette_dataset("../../Datasets/imagenette2", train_trans=trans, val_trans=val_trans)
    train_classifier(train_set, val_set)
    # CIAR10 and MNIST are already trained :D

