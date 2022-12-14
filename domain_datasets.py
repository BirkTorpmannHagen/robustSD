import os
import json
import torch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from glob import glob
from torch.utils import data
import torchvision.transforms as transforms


class NICODataset(data.Dataset):
    def __init__(self, image_path_list, label_map_json, transform):
        super().__init__()
        self.image_path_list = image_path_list
        self.transform = transform
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


def build_nico_dataset(use_track, root, val_ratio, train_transform, val_transform, context, seed=0):
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

    np.random.RandomState(seed).shuffle(image_path_list)
    n = round((len(image_path_list) * val_ratio) / 2) * 2
    train_dataset = NICODataset(image_path_list[n:], label_map_json, train_transform)
    val_dataset = NICODataset(image_path_list[:n], label_map_json, val_transform)
    return train_dataset, val_dataset

def build_polyp_dataset(root, val_ratio, train_transform, val_transform, fold, seed=0):
    pass

class KvasirSegmentationDataset(data.Dataset):
    """
        Dataset class that fetches images with the associated segmentation mask.
    """
    def __init__(self, path, split="train"):
        super(KvasirSegmentationDataset, self).__init__()
        self.path = join(path, "segmented-images/")
        self.fnames = listdir(join(self.path, "images"))
        self.common_transforms = pipeline_tranforms()
        self.split = split
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

        image = np.array(open(join(self.path, "images/", self.split_fnames[index])).convert("RGB"))
        mask = np.array(open(join(self.path, "masks/", self.split_fnames[index])).convert("L"))
        image = self.common_transforms(PIL.Image.fromarray(image))
        mask = self.common_transforms(PIL.Image.fromarray(mask))
        mask = (mask > 0.5).float()
        return image, mask, self.split_fnames[index]
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
    for x, y, context in build_nico_dataset(1, "../../Datasets/NICO++", 0, trans, trans, context="autumn", seed=0)[0]:
        print(context)