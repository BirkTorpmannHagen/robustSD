import pandas as pd
def convert(fname):
    data = pd.read_csv(fname)
    data = data.drop(columns=["Unnamed: 0", "ind_dataset", "rep_model"]).rename(columns={"ood_dataset": "fold"}).replace("polyp_ood", "ood").replace("polyp_ind", "ind")
    data
    data_knn = data.drop(columns=["vanilla_p"]).rename(columns={"kn_p": "pvalue"})
    data_vanilla = data.drop(columns=["kn_p"]).rename(columns={"vanilla_p": "pvalue"})
    for sample_size in pd.unique(data["sample_size"]):
        data_knn[data_knn["sample_size"] == sample_size].to_csv(f"Polyp_ks_5NN_{sample_size}.csv")
        data_vanilla[data_vanilla["sample_size"] == sample_size].to_csv(f"Polyp_ks_{sample_size}.csv")

    print(data_knn.groupby(["fold", "sampler"]).mean())


if __name__ == '__main__':
    convert("lp_data_polyps.csv")