from segmentor.deeplab import SegmentationModel
from pytorch_lightning import Trainer
from domain_datasets import build_nico_dataset, build_polyp_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import warnings
import torch
from torch.utils.data import ConcatDataset
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')

#

def train_segmentor():
    import os
    model = SegmentationModel(transfer=False, batch_size=16)
    # model = SegmentationModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_4/checkpoints/epoch=199-step=20000.ckpt", resnet_version=34)
    logger = TensorBoardLogger(save_dir="segmentation_logs")
    trainer = Trainer(accelerator="gpu", max_epochs=200,logger=logger,num_processes=1)
    # trans = transforms.Compose([
    #                     transforms.Resize((512,512)),
    #                     transforms.ToTensor(), ])
    ind, val, ood = build_polyp_dataset("../../Datasets/Polyps")
    train_loader = DataLoader(ind, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=16, shuffle=True, num_workers=4)
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)

if __name__ == '__main__':
    train_segmentor()

