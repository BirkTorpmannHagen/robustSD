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
    # model = SegmentationModel(34)
    model = SegmentationModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_4/checkpoints/epoch=199-step=20000.ckpt", resnet_version=34)
    logger = TensorBoardLogger(save_dir="segmentation_logs")
    trainer = Trainer(accelerator="gpu", devices="cuda:0", max_epochs=200,logger=logger,num_processes=1)
    trans = transforms.Compose([
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    cvc_train_set, cvc_val_set = build_polyp_dataset("../../Datasets/Polyps/CVC-ClinicDB", "CVC", 0)
    kvasir_train_set, kvasir_val_set = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    train_set = ConcatDataset((cvc_train_set, kvasir_train_set))
    val_set = ConcatDataset((cvc_val_set, kvasir_val_set))


    trainer.fit(model, train_dataloaders=DataLoader(train_set, shuffle=True, num_workers=4, batch_size=8,drop_last=True),val_dataloaders=DataLoader(val_set, shuffle=True, num_workers=4, batch_size=2))

if __name__ == '__main__':
    train_segmentor()

