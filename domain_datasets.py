import os
import json
import albumentations as alb
import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset
import numpy as np
from PIL import Image
from glob import glob
from torchvision.transforms import ToTensor
from torch.utils import data
import torchvision.transforms as transforms
from os import listdir
import torchvision
from os.path import join
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder
from njord.utils.general import check_dataset
from njord.utils.dataloaders import create_dataset, create_dataloader
from random import shuffle

class Pneumonia(data.Dataset):
    def __init__(self, root, train_trans, val_trans, fold):

        if fold=="ood":
            self.dataset = data.ConcatDataset([ImageFolder(join(root, "PneumoniaWomen", "train"), val_trans),
                                               ImageFolder(join(root, "PneumoniaWomen", "test"), val_trans),
                                               ImageFolder(join(root, "PneumoniaWomen", "val"), val_trans)])
        else:
            if fold=="train":
                self.dataset = ImageFolder(join(root, "PediatricPneumonia", fold), train_trans)
            else:
                self.dataset = ImageFolder(join(root, "PneumoniaWomen", "test"), val_trans)
        self.num_classes = 2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        y = torch.zeros(2)
        y[self.dataset[item][1]] = 1
        return self.dataset[item][0], y

def get_pneumonia_dataset(root, train_trans, val_trans):
    train = Pneumonia(root, train_trans, val_trans, "train")
    val = Pneumonia(root, train_trans, val_trans, "val")
    test = Pneumonia(root, train_trans, val_trans, "test")
    ood = Pneumonia(root, train_trans, val_trans, "ood")
    return train, val, test, ood

class KvasirSegmentationDataset(data.Dataset):
    """
        Dataset class that fetches images with the associated segmentation mask.
    """
    def __init__(self, path, train_alb, val_alb, split="train"):
        super(KvasirSegmentationDataset, self).__init__()
        self.path = path
        self.fnames = listdir(join(self.path,"segmented-images", "images"))
        self.split = split
        self.train_transforms = train_alb
        self.val_transforms = val_alb
        train_size = int(len(self.fnames) * 0.8)
        val_size = (len(self.fnames) - train_size) // 2
        test_size = len(self.fnames) - train_size - val_size
        self.fnames_train = self.fnames[:train_size]
        self.fnames_val = self.fnames[train_size:train_size + val_size]
        self.fnames_test = self.fnames[train_size + val_size:]
        self.split_fnames = None  # iterable for selected split
        if self.split == "train":
            self.size = train_size
            self.split_fnames = self.fnames_train
        elif self.split == "val":
            self.size = val_size
            self.split_fnames = self.fnames_val
        elif self.split == "test":
            self.size = test_size
            self.split_fnames = self.fnames_test
        else:
            raise ValueError("Choices are train/val/test")
        self.tensor = ToTensor()


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # img = Image.open(join(self.path, "segmented-images", "images/", self.split_fnames[index]))
        # mask = Image.open(join(self.path, "segmented-images", "masks/", self.split_fnames[index]))

        image = np.asarray(Image.open(join(self.path, "segmented-images", "images/", self.split_fnames[index])))
        mask =  np.asarray(Image.open(join(self.path, "segmented-images", "masks/", self.split_fnames[index])))
        if self.split=="train":
            image, mask = self.train_transforms(image=image, mask=mask).values()
        else:
            image, mask = self.val_transforms(image=image, mask=mask).values()
        image, mask = transforms.ToTensor()(Image.fromarray(image)), transforms.ToTensor()(Image.fromarray(mask))
        mask = torch.mean(mask,dim=0,keepdim=True).int()
        return image,mask

class Imagenette(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, split="train"):
        super().__init__(os.path.join(root, split), transform=transform)
        self.num_classes = 10


class NICODataset(data.Dataset):
    def __init__(self, image_path_list, label_map_json, transform):
        super().__init__()
        self.image_path_list = image_path_list
        self.transform = transform
        self.num_classes = len(os.listdir(os.path.join(*image_path_list[0].split("/")[:-2])))
        self.classes = os.listdir(os.path.join(*image_path_list[0].split("/")[:-2]))
        with open(label_map_json, "r") as f:
            self.label_map = json.load(f)
        context_path = os.path.join(*image_path_list[0].split("/")[:-3])
        self.context = image_path_list[0].split("/")[-3]
        contexts = os.listdir(context_path)
        self.context_map = dict(zip(contexts, range(len(contexts))))
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path)
        image = self.transform(image)
        label = self._get_label_index(image_path)
        return image, label, self.context_map[image_path.split("/")[-3]]

    def _get_label_index(self, image_path):
        class_name = image_path.split("/")[-2]
        label_index = self.label_map[class_name]
        return label_index


    def fetch_model(self):
        """
        :return: trained classifier, pytorch lightning
        """
        pass


def get_njordvid_datasets():
    ind_data_dict = check_dataset("njord/folds/ind_fold.yaml")
    ind_train_path, ind_val_path = ind_data_dict['train'], ind_data_dict['val']
    ind_dataloader, _ = create_dataloader(ind_train_path, 512,1,32, image_weights=True)
    ind_val_dataloader = create_dataloader(ind_val_path, 512,1,32, image_weights=True)

    ood_data_dict = check_dataset("njord/folds/ood_fold.yaml")
    _, ood_val_path = ood_data_dict['train'], ood_data_dict['val']
    ood_dataloader, _= create_dataloader(ood_val_path, 512, 1, 32, image_weights=True)

    return ind_dataloader, ind_val_dataloader, ood_dataloader

# def get_cifar10_datase
class EtisDataset(data.Dataset):
    """
        Dataset class that fetches Etis-LaribPolypDB images with the associated segmentation mask.
        Used for testing.
    """

    def __init__(self, path, val_alb, split="train"):
        super(EtisDataset, self).__init__()
        self.path = path
        self.len = len(listdir(join(self.path, "Original")))
        indeces = range(self.len)
        self.train_indeces = indeces[:int(0.8*self.len)]
        self.val_indeces = indeces[int(0.8*self.len):]
        self.transforms = val_alb
        self.split = split
        if self.split=="train":
            self.len=len(self.train_indeces)
        else:
            self.len=len(self.val_indeces)
        self.tensor = ToTensor()

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        if self.split=="train":
            index = self.train_indeces[i]
        else:
            index = self.val_indeces[i]


        img_path = join(self.path, "Original/{}.jpg".format(index + 1))
        mask_path = join(self.path, "GroundTruth/p{}.jpg".format(index + 1))
        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        image, mask = self.transforms(image=image, mask=mask).values()

        return self.tensor(image), self.tensor(mask)[0].unsqueeze(0).int()

class CVC_ClinicDB(data.Dataset):
    def __init__(self, path, transforms, split="train"):
        super(CVC_ClinicDB, self).__init__()
        self.path = path
        self.len = len(listdir(join(self.path, "Original")))
        indeces = range(self.len)
        self.train_indeces = indeces[:int(0.8*self.len)]
        self.val_indeces = indeces[int(0.8*self.len):]
        self.transforms = transforms
        self.split = split
        if self.split=="train":
            self.len=len(self.train_indeces)
        else:
            self.len=len(self.val_indeces)
        self.common_transforms = transforms
        self.tensor = ToTensor()

    def __getitem__(self, i):
        if self.split=="train":
            index = self.train_indeces[i]
        else:
            index = self.val_indeces[i]


        img_path = join(self.path, "Original/{}.png".format(index + 1))
        mask_path = join(self.path, "Ground Truth/{}.png".format(index + 1))
        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        image, mask = self.transforms(image=image, mask=mask).values()
        # mask = (mask>0.5).int()[0].unsqueeze(0)
        return self.tensor(image), self.tensor(mask)[0].unsqueeze(0).int()

    def __len__(self):
        return self.len

class ImagenettewNoise(Imagenette):
    def __init__(self, root,transform,train, noise_level=0):
        super().__init__(root, transform, train)
        self.noise_level = noise_level

    def __getitem__(self, index):
        x,y = super().__getitem__(index)
        if self.noise_level!=0:
            x = torch.clip(x + torch.randn_like(x)*self.noise_level, 0, 1)
        return x,y

    def __len__(self):
        return super().__len__()

class CIFAR10wNoise(CIFAR10):
    def __init__(self, root, train, transform, noise_level=0, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.noise_level = noise_level
        self.plotted = False

    def __getitem__(self, index):
        x,y = super().__getitem__(index)
        if self.noise_level!=0:
            x = torch.clip(x + torch.randn_like(x)*self.noise_level, 0, 1)
        return x,y

    def __len__(self):
        return super().__len__()

class CIFAR100wNoise(CIFAR100):
    def __init__(self, root, train, transform, noise_level=0, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.noise_level = noise_level
        self.plotted = False

    def __getitem__(self, index):
        x,y = super().__getitem__(index)
        if self.noise_level!=0:
            x = torch.clip(x + torch.randn_like(x)*self.noise_level, 0, 1)
        return x,y

    def __len__(self):
        return super().__len__()


def build_nico_dataset(use_track, root, val_ratio, train_transform, val_transform, context, seed=0):
    if use_track == 1:
        track_data_dir = os.path.join(root, "track_1")
        data_dir = os.path.join(track_data_dir, "public_dg_0416", "train")
        label_map_json = os.path.join(track_data_dir, "dg_label_id_mapping.json")
        image_path_list = glob(f"{data_dir}/{context}/*/*.jpg")
        shuffle(image_path_list)

    else:
        track_data_dir = os.path.join(root, "track_2")
        data_dir = os.path.join(
            track_data_dir, "public_ood_0412_nodomainlabel", "train"
        )
        label_map_json = os.path.join(track_data_dir, "ood_label_id_mapping.json")
        image_path_list = glob(f"{data_dir}/*/*.jpg")
        shuffle(image_path_list)

    if val_ratio==0:
        return NICODataset(image_path_list, label_map_json, train_transform), NICODataset(image_path_list, label_map_json, train_transform)

    n = round((len(image_path_list) * val_ratio) / 2) * 2
    train_dataset = NICODataset(image_path_list[n:], label_map_json, train_transform)
    val_dataset = NICODataset(image_path_list[:n], label_map_json, val_transform)
    return train_dataset, val_dataset




def build_polyp_dataset(root):
    translist = [alb.Compose([
        i,
        alb.Resize(512, 512)]) for i in [alb.HorizontalFlip(p=0), alb.HorizontalFlip(always_apply=True),
                                         alb.VerticalFlip(always_apply=True), alb.RandomRotate90(always_apply=True),
                                         ]]
    inds = []
    vals = []
    oods = []
    for trans in translist:
        cvc_train_set = CVC_ClinicDB(join(root, "CVC-ClinicDB"),trans, split="train")
        cvc_val_set = CVC_ClinicDB(join(root, "CVC-ClinicDB"),trans, split="val")
        kvasir_train_set = KvasirSegmentationDataset(join(root, "HyperKvasir"), train_alb=trans, val_alb=trans)
        kvasir_val_set = KvasirSegmentationDataset(join(root, "HyperKvasir"), train_alb=trans, val_alb=trans, split="val")
        etis_train_set = EtisDataset(join(root, "ETIS-LaribPolypDB"), trans, split="train")
        etis_val_set = EtisDataset(join(root, "ETIS-LaribPolypDB"), trans, split="val")

        inds.append(kvasir_train_set)
        inds.append(cvc_train_set)

        vals.append(kvasir_val_set)
        vals.append(cvc_val_set)

        oods.append(etis_train_set)
        oods.append(etis_val_set)

    ind = ConcatDataset(inds)
    ind_val = ConcatDataset(vals)
    ood = ConcatDataset(oods)
    return ind, ind_val, ood

def build_njord_datasets():


    ind = check_dataset("njord/folds/ind_fold.yaml")
    ood = check_dataset("njord/folds/ood_fold.yaml")

    train_set = create_dataset(ind["train"], 512, 16, 32)
    val_set =  create_dataset(ind["val"], 512, 16, 32)
    ood_set =  create_dataset(ood["val"], 512, 16, 32)
    return train_set, val_set, ood_set



class NICOTestDataset(data.Dataset):
    def __init__(self, image_path_list, transform):
        super().__init__()
        self.image_path_list = image_path_list
        self.transform = transform

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path)
        image = self.transform(image)
        return image


def build_imagenette_dataset(root, train_trans, val_trans):
    train = Imagenette(root, transform=train_trans, split="train")
    val = Imagenette(root, transform=val_trans, split="val")
    return train, val


if __name__ == '__main__':
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              # transforms.CenterCrop(148), #2048 with, 4096 without...
                                              transforms.Resize((256,256)),
                                              transforms.ToTensor(),])
    train, val= build_imagenette_dataset("../../Datasets/imagenette2", trans)
