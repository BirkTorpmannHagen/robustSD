import pandas as pd
import os
from os.path import join
from shutil import copy
ROOT = "../../Datasets/NjordYolo/"
if __name__ == '__main__':
    new_ind = list(pd.read_csv("new_ind_njord.csv")["path"].apply(lambda x: x.split("/")[-1][:-3]))
    new_ood = list(pd.read_csv("new_ood_njord.csv")["path"].apply(lambda x: x.split("/")[-1][:-3]))
    # os.mkdir("../../Datasets/NjordYolo/ind")
    # os.mkdir("../../Datasets/NjordYolo/ood")
    # os.mkdir("../../Datasets/NjordYolo/ood/images")
    # os.mkdir("../../Datasets/NjordYolo/ood/labels")
    # os.mkdir("../../Datasets/NjordYolo/ind/images")
    # os.mkdir("../../Datasets/NjordYolo/ind/labels")
    ind_img_path = "../../Datasets/NjordYolo/ind/images"
    ind_label_path = "../../Datasets/NjordYolo/ind/labels"
    ood_img_path = "../../Datasets/NjordYolo/ood/images"
    ood_label_path = "../../Datasets/NjordYolo/ood/labels"
    for day in os.listdir(ROOT):
        if day=="ind" or day=="ood":
            continue
        for file in os.listdir(join(ROOT, day, "images")):
            label_name = file.replace(".jpg", ".txt")
            if file in new_ind:
                copy(join(ROOT,day, "images", file), join(ind_img_path, file))
                try:
                    copy(join(ROOT,day, "labels", label_name), join(ind_label_path, label_name))
                except FileNotFoundError:
                    pass
            elif file in new_ood:
                copy(join(ROOT,day, "images", file), join(ood_img_path, file))
                try:
                    copy(join(ROOT,day, "labels", label_name), join(ood_label_path, label_name))
                except FileNotFoundError:
                    pass


