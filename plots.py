import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from metrics import *


if __name__ == '__main__':
    dataset = pd.read_csv("data.csv")
    for sample_size in [10, 20, 50, 100, 200, 500, 1000, 10000]:
        subset = dataset[dataset["Sample Size"]==sample_size]
        kn = subset[subset["Method"]!="Vanilla"]
        vanilla = subset[subset["Method"]=="Vanilla"]
        kn_ind = kn[kn["Fold"]=="dim"]
        vanilla_ind = vanilla[vanilla["Fold"]=="dim"]
        kn_ood = kn[kn["Fold"]!="dim"]
        vanilla_ood = vanilla[vanilla["Fold"]!="dim"]
        # vanilla_ood.loc[:, "P"]*=256
        # vanilla_ind.loc[:,"P"]*=256
        print(sample_size)
        # correlation(kn_ood, kn_ind)
        # correlation(vanilla_ood, vanilla_ind)
        # if sample_size>500:
        #     sns.regplot(data=kn, x="P", y="loss")
        #     plt.show()
        # input()
        # print(sample_size)
        # fpr_van = fprat95tpr(vanilla_ood, vanilla_ind)
        # fpr_kn = fprat95tpr(kn_ood, kn_ind)
        # print("fpr95tpr vanilla: ", fpr_van)
        # print("fpr95tpr kn: ", fpr_kn)
        # # plt.hist(kn_ood["P"], label="ood")
        # # plt.hist(kn_ind["P"], label="ind")
        # # plt.title("kn")
        # # plt.legend()
        # # plt.show()
        # # plt.hist(vanilla_ood["P"], label="ood")
        # # plt.hist(vanilla_ind["P"], label="ind")
        # # plt.title("vanilla")
        # # plt.legend()
        # # plt.show()
        auc_van = auroc(vanilla_ood, vanilla_ind)
        print("vanilla auc: ", auc_van)
        auc_kn = auroc(kn_ood, kn_ind)
        print("kn auc: ", auc_kn)
        # auc_van = aupr(vanilla_ood, vanilla_ind)
        # print("vanilla aupr: ", auc_van)
        # auc_kn = aupr(kn_ood, kn_ind)
        # print("kn aupr: ", auc_kn)