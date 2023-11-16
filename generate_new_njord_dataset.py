import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import join
from shutil import copy
import seaborn as sns
ROOT = "/home/birk/Datasets/NjordYolo"

def generate_new_spltis():
    df = pd.read_csv("merged_njord_loses.csv")
    new_df = df[df["loss"]!=0]
    new_ind = new_df[df["loss"]<=0.8]
    new_ood = new_df[df["loss"]>0.8]
    new_ind.to_csv("new_ind_njord.csv")
    new_ood.to_csv("new_ood_njord.csv")

    sns.kdeplot(data=new_df, x="loss", hue="fold")
    plt.show()

def create_data():
    new_ind = list(pd.read_csv("new_ind_njord.csv")["path"].apply(lambda x: x.split("/")[-1][:-3]))
    new_ood = list(pd.read_csv("new_ood_njord.csv")["path"].apply(lambda x: x.split("/")[-1][:-3]))
    os.mkdir(f"{ROOT}/ind")
    os.mkdir(f"{ROOT}/ood")
    os.mkdir(f"{ROOT}/ood/images")
    os.mkdir(f"{ROOT}/ood/labels")
    os.mkdir(f"{ROOT}/ind/images")
    os.mkdir(f"{ROOT}/ind/labels")
    ind_img_path = f"{ROOT}/ind/images"
    ind_label_path = f"{ROOT}/ind/labels"
    ood_img_path = f"{ROOT}/ood/images"
    ood_label_path = f"{ROOT}/ood/labels"
    for day in os.listdir(ROOT):
        print(day)
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

if __name__ == '__main__':
    generate_new_spltis()
    create_data()



