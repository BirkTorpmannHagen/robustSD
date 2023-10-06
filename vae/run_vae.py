import os
import yaml
import argparse
from pathlib import Path
from models.vanilla_vae import VanillaVAE, ResNetVAE, CIFARVAE
from vae_experiment import VAEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from domain_datasets import *
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


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='vae/configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)



model = ResNetVAE()
# model = CIFARVAE()
experiment = VAEXperiment(model,
                          config['exp_params'])

# train, val, test = get_njordvid_datasets()
# from torch.utils.data import DataLoader


# train = CIFAR10("../../Datasets/CIFAR10", train=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((32,32))]), download=True)
# val = CIFAR10("../../Datasets/CIFAR10", train=False, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((32,32))]), download=True)



train = CIFAR100("../../Datasets/CIFAR100", train=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((32,32))]), download=True)
val = CIFAR100("../../Datasets/CIFAR100", train=False, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((32,32))]), download=True)
tb_logger =  TensorBoardLogger(save_dir="vae_logs",
                               name="CIFAR100")
data = VAEDataset(**config["data_params"], train_set=train, val_set=val)

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


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)