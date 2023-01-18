from segmentor.deeplab import SegmentationModel
from pytorch_lightning import Trainer
from domain_datasets import build_nico_dataset, build_polyp_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import warnings
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')

#

def train_segmentor():
    import os
    model = SegmentationModel(34)
    logger = TensorBoardLogger(save_dir="segmentation_logs")
    trainer = Trainer(accelerator="gpu", devices="cuda:0", max_epochs=200,logger=logger,num_processes=1)
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    train_set, val_set = build_polyp_dataset("../../Datasets/Polyps/CVC-ClinicDB", "CVC", 0)
    trainer.fit(model, train_dataloaders=DataLoader(train_set, shuffle=True, num_workers=4, batch_size=16),val_dataloaders=DataLoader(val_set, shuffle=True, num_workers=4))

if __name__ == '__main__':
    train_segmentor()

