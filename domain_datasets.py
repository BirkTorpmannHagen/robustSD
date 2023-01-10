import os
import json
import torch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from glob import glob
from torch.utils import data
import torchvision.transforms as transforms
from os import listdir
from os.path import join
import albumentations as alb

class KvasirSegmentationDataset(data.Dataset):
    """
        Dataset class that fetches images with the associated segmentation mask.
    """
    def __init__(self, path, split="train"):
        super(KvasirSegmentationDataset, self).__init__()
        self.path = path
        self.fnames = listdir(join(self.path,"segmented-images", "images"))
        self.split = split
        self.train_transforms = alb.Compose([alb.Flip(), alb.Resize(512,512)])
        self.val_transforms = alb.Compose([alb.Resize(512,512)])
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
        return image,mask, "Kvasir"
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

class NicoClassBiasedDataset(NICODataset):
    """
        Fetches samples from one given class
    """
    def __init__(self, image_path_list, label_map_json, transform, class_index):
        super().__init__(image_path_list, label_map_json, transform)
        class_weights = [len(list(filter(lambda x: x.split("/")[-2]==j, image_path_list)))/len(self) for j in self.classes]
        classwise_paths = [list(filter(lambda x: x.split("/")[-2]==j, image_path_list)) for j in self.classes]
        # self.classwise_paths_flat = sum(classwise_paths, []) #if iterative bias
        self.classwise_paths_flat = classwise_paths[class_index]
        print(self.classwise_paths_flat)

    def __getitem__(self, item):
        return self.classwise_paths_flat[item]

    def __len__(self):
        return len(self.classwise_paths_flat)

class NjordVideoBiasDataset(data.Dataset):
    """
    Samples selected according to recency wrt frames
    """
    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass




def build_nico_dataset(use_track, root, val_ratio, train_transform, val_transform, context, biased=False, seed=0):
    if use_track == 1:
        track_data_dir = os.path.join(root, "track_1")
        data_dir = os.path.join(track_data_dir, "public_dg_0416", "train")
        label_map_json = os.path.join(track_data_dir, "dg_label_id_mapping.json")
        image_path_list = glob(f"{data_dir}/{context}/*/*.jpg")
    else:
        track_data_dir = os.path.join(root, "track_2")
        data_dir = os.path.join(
            track_data_dir, "public_ood_0412_nodomainlabel", "train"
        )
        label_map_json = os.path.join(track_data_dir, "ood_label_id_mapping.json")
        image_path_list = glob(f"{data_dir}/*/*.jpg")
    if val_ratio==0:
        return NICODataset(image_path_list, label_map_json, train_transform), NICODataset(image_path_list, label_map_json, train_transform)

    np.random.RandomState(seed).shuffle(image_path_list) # shuffles. Perhaps a bad idea to do at dataset level
    n = round((len(image_path_list) * val_ratio) / 2) * 2
    if not biased:
        train_dataset = NICODataset(image_path_list[n:], label_map_json, train_transform)
        val_dataset = NICODataset(image_path_list[:n], label_map_json, val_transform)
    else:
        print("biased!")
        train_dataset = NicoClassBiasedDataset(image_path_list[n:], label_map_json, train_transform)
        val_dataset = NicoClassBiasedDataset(image_path_list[:n], label_map_json, val_transform)
    return train_dataset, val_dataset

def build_polyp_dataset(root, fold, seed=0):
    if fold=="Kvasir":
        train_set = KvasirSegmentationDataset(root, split="train")
        val_set = KvasirSegmentationDataset(root, split="val")
    else:
        raise NotImplementedError
    return train_set, val_set

def build_njord_dataset():
    pass

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




if __name__ == '__main__':
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              # transforms.CenterCrop(148), #2048 with, 4096 without...
                                              transforms.Resize(512),
                                              transforms.ToTensor(),])
    build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="autumn", biased=True, seed=0)[0]