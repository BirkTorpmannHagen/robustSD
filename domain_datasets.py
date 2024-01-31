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
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST
from torchvision.datasets import ImageFolder
from njord.utils.general import check_dataset
from njord.utils.dataloaders import LoadImagesAndLabels
from random import shuffle
import albumentations
def additive_noise(x, intensity):
    noise = torch.randn_like(x) * intensity
    return x + noise

def targeted_fgsm(model, x, intensity):
    #adversarial attack to generate high-confidence false predictions
    input_sample = x.clone().detach().requires_grad_(True)
    with torch.no_grad():
        output = model(input_sample)
    target_label = torch.zeros_like(output)
    target_label[:, torch.randint(0, output.shape[1], (1,))] = 1

    for _ in range(5):
        output = model(input_sample)
        loss = torch.nn.CrossEntropyLoss()(output, target_label)

        model.zero_grad()
        loss.backward()

        # Apply perturbation
        input_sample.data = input_sample.data - intensity * input_sample.grad.sign()

        # Check if the sample has crossed the decision boundary
        if model(input_sample).argmax(1) == target_label:
            break

    return input_sample

def random_occlusion(x, intensity):
    img_size = max(x.shape)
    occlusion = albumentations.Cutout(8, max_h_size=min(x//16, 8))

class TransformedDataset(data.Dataset):
    #generic wrapper for adding noise to datasets
    def __init__(self, dataset, transform, transform_name, transform_param):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.transform_param = transform_param
        self.transform_name = transform_name
        print(transform_name)
        print(transform_param)
    def __getitem__(self, index):

        batch = self.dataset.__getitem__(index)
        x = batch[0]
        rest = batch[1:]
        x = torch.clip(self.transform(x, self.transform_param), 0, 1)
        if index==0:
            plt.imshow(x.permute(1,2,0))
            plt.savefig(f"test_plots/{self.transform_name}_{self.transform_param}.png")
            plt.show()
            plt.close()
        return (x, *rest)

    def __str__(self):
        return f"{type(self.dataset).__name__}_{self.transform_name}_{str(self.transform_param)}"

    def __len__(self):
        # return 1000 #debug
        return self.dataset.__len__()

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
        # return 1000 #debug
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

def build_polyp_dataset(root, ex=False):
    if ex:
        translist = [alb.Compose([
            i,
            alb.Resize(512, 512)]) for i in [alb.HorizontalFlip(p=0), alb.HorizontalFlip(always_apply=True),
                                             alb.VerticalFlip(always_apply=True), alb.RandomRotate90(always_apply=True),
                                             ]]
    else:
        translist = [alb.Compose([
            alb.Resize(512, 512)])]
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

def create_dataset(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False,
                      natively_trainable=False):
    dataset = NjordDataset(
        path,
        imgsz,
        batch_size,
        augment=augment,  # augmentation
        hyp=hyp,  # hyperparameters
        rect=rect,  # rectangular batches
        cache_images=cache,
        single_cls=single_cls,
        stride=int(stride),
        pad=pad,
        image_weights=image_weights,
        prefix=prefix,
        natively_trainable=natively_trainable)
    return dataset
def build_njord_datasets():


    ind = check_dataset("njord/folds/ind_fold.yaml")
    ood = check_dataset("njord/folds/ood_fold.yaml")

    train_set = create_dataset(ind["train"], 512, 16, 32,natively_trainable=True)
    val_set =  create_dataset(ind["val"], 512, 16, 32, natively_trainable=True)
    ood_set =  create_dataset(ood["val"], 512, 16, 32, natively_trainable=True)
    return train_set, val_set, ood_set


class NjordDataset(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 prefix='',
                 natively_trainable=False):
        super().__init__(
                 path,
                 img_size,
                 batch_size,
                 augment,
                 hyp,
                 rect,
                 image_weights,
                 cache_images,
                 single_cls,stride,pad,prefix, natively_trainable)

    def __getitem__(self, item):
        x, targets, paths, shapes = super().__getitem__(item)
        return x, targets, paths, shapes
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

class MNIST3(MNIST):
    def __init__(self, root, train, transform, download=False):
        self.num_classes = 10
        super().__init__(root, train, transform, download=download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img.repeat(3,1,1), target

class EMNIST3(EMNIST):
    def __init__(self, root, train, transform, download=False):
        super().__init__(root, "letters", train=train, transform=transform, download=download)
        self.num_classes = 27
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img.repeat(3,1,1), target

if __name__ == '__main__':
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              # transforms.CenterCrop(148), #2048 with, 4096 without...
                                              transforms.Resize((32,32)),
                                              transforms.ToTensor(),])
    train, val= build_imagenette_dataset("../../Datasets/imagenette2", trans)
