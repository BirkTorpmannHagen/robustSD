import os
import yaml
import argparse
from pathlib import Path
from models.vanilla_vae import VanillaVAE, ResNetVAE, CIFARVAE
from vae_experiment import VAEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from domain_datasets import *
from torch.utils.data import DataLoader
# from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning_dataset import VAEDataset
from torchvision.datasets import CIFAR10, CIFAR100
# from pytorch_lightning.plugins import DDPPlugin

def wrap_dataset(dataset):
    """
    dumb utility function to make testing easier. standardizes datasets so that it works easier with the models and trainers
    :param dataset:
    :return:
    """
    class NewDataset(data.Dataset):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

        def __getitem__(self, index):
            image, label = self.dataset[index]
            if image.shape[0]==1:
                image = image.repeat(3,1,1)
            return image, label, 0

        def __len__(self):
            return len(self.dataset)

    return NewDataset(dataset)


patch_size = 512
model = ResNetVAE(patch_size=patch_size)
# model = CIFARVAE()
params = {
  "LR": 0.00005,
  "weight_decay": 0.0,
  "scheduler_gamma": 0.95,
  "kld_weight": 0.00025,
  "manual_seed": 1265

}
experiment = VAEXperiment(model, params)

# train_trans= transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((256,256)), transforms.ToTensor()])
# val_trans= transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((256,256)), transforms.ToTensor()])
# train, val = build_imagenette_dataset("../../Datasets/imagenette2", train_trans, val_trans)

# train, val = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", trans=alb.Compose([alb.RandomRotate90(), alb.Flip(), alb.Resize(patch_size, patch_size)]), fold="Kvasir")
train, val, ood = get_njordvid_datasets()
tb_logger =  TensorBoardLogger(save_dir="vae_logs",
                               name=train.__class__.__name__)
data = VAEDataset(train_set=train, val_set=val)

data.setup()
runner = Trainer(logger=tb_logger,
                 accelerator="gpu",
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=3,
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training =======")
runner.fit(experiment, datamodule=data)