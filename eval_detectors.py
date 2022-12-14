import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vae.vae_experiment import VAEXperiment
from vae.models.vanilla_vae import VanillaVAE
import yaml
import torch
from torch.utils.data import DataLoader
from yellowbrick.features.manifold import Manifold
from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle as pkl
import torch.utils.data as data
from domain_datasets import build_nico_dataset, build_polyp_dataset
from classifier.resnetclassifier import ResNetClassifier

def transform_dataset(dataset, transform):
    class NewDataset(data.Dataset):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

        def __getitem__(self, index):
            image, label, context = self.dataset[index]
            image = transform(image)
            return image, label, context

        def __len__(self):
            return len(self.dataset)

    return NewDataset(dataset)

def get_results(organic_benches, synthetic_benches):
    noise = [lambda x: x+torch.randn_like(x)*i for i in np.linspace(0, 0.5, 10)]
    for dataset in organic_benches:
        for x,y,_ in dataset:
            pass
    for synthetic_dataset in synthetic_benches:
        for noise_transform in noise:
            bench = transform_dataset(synthetic_dataset, noise_transform)
            for x,y,_ in bench:
                bench.eval_sample(x,y)

class robustSD:
    def __init__(self, rep_model, classifier, config):
        self.rep_model = rep_model
        self.classifier = classifier
        self.config = config


    def compute_pvals_and_loss(self, ind_dataset, ood_dataset, ood_sampler, sample_size, ind_dataset_name, ood_dataset_name, plot=False):
        # todo make this neater by incorporating the Bench classes
        sample_size = min(sample_size, len(ood_dataset))
        fname_encodings = f"robustSD_{ind_dataset_name}_enodings_{self.rep_model.__name__}.pkl"
        fname_losses = f"robustSD_{ind_dataset_name}_losses_{self.rep_model.__name__}.pkl"
        try:
            ind_latents = pkl.load(open(fname_encodings, "rb"))
            losses = pkl.load(open(fname_losses, "rb"))
        except FileNotFoundError:
            ind_latents = np.zeros((len(ind_dataset), self.rep_model.latent_dim))
            losses = np.zeros(len(ind_dataset))
            for i, (x, y, _) in tqdm(enumerate(DataLoader(ind_dataset)),total=len(ind_dataset)):
                with torch.no_grad():
                    ind_latents[i] = self.rep_model.encode(x.to(self.config["device"]))[0].cpu().numpy()
                    losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),y.to(self.config["device"])).cpu().numpy()
            pkl.dump(ind_latents, open(fname_encodings, "wb"))
            pkl.dump(losses, open(fname_losses, "wb"))

        ood_latents = np.zeros((len(ood_dataset), self.rep_model.latent_dim))
        ood_losses = np.zeros(len(ood_dataset))
        for i, (x, y, _) in tqdm(enumerate(DataLoader(ood_dataset, sampler=ood_sampler)), total=len(ood_dataset)):
            with torch.no_grad():
                ood_latents[i] = self.rep_model.encode(x.to(self.config["device"]))[0].cpu().numpy()

                ood_losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),
                                                         y.to(self.config["device"])).cpu().numpy()
        cols = ["ind_dataset", "ood_dataset", "vanilla_p", "kn_p", "loss"]
        dataframe = pd.DataFrame(columns=cols)

        for i, j in zip(range(0, len(ind_dataset), sample_size), range(sample_size, len(ind_dataset)+sample_size, sample_size))[:-1]: #perform tests
            sample_idx = range(i,j)
            ood_samples = ood_latents[sample_idx]

            vanilla_pval = min(np.min([ks_2samp(ind_latents[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)]) * self.rep_model.latent_dim, 1)

            k_nearest_ind = [np.argmin(np.sum((np.expand_dims(i, 0) - ind_latents) ** 2, axis=-1)) for i in ood_samples]
            kn_pval = min(np.min([ks_2samp(k_nearest_ind[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)]) * self.rep_model.latent_dim, 1)
            dataframe.append(dict(zip(cols, [ind_dataset_name, ood_dataset_name, vanilla_pval, kn_pval, np.mean(ood_losses[sample_idx])])), ignore_index=True)
        return dataframe


    def bootstrap_severity_estimation(self):
        pass

    def eval_synthetic(self, ind_dataset, transforms, sample_size, plot=False):
        datasets = [transform_dataset(ind_dataset, transform) for transform in transforms]
        results = []
        for dataset in datasets:
            results.append(self.compute_pvals_and_loss(ind_dataset, dataset, sample_size, plot=plot))
        return results





if __name__ == '__main__':
    # generate_plot(create_datasets_by_fold(), ["ind", "test_val"])
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))

    ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
    ood_1 = build_nico_dataset(1, "../../Datasets/NICO++", 0.1, trans, trans, context="rock", seed=0)[1]
    ood_2 = build_nico_dataset(1, "../../Datasets/NICO++", 0.1, trans, trans, context="autumn", seed=0)[1]
    ood_3 = build_nico_dataset(1, "../../Datasets/NICO++", 0.1, trans, trans, context="water", seed=0)[1]

    # ind, ind_val = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    config = yaml.safe_load(open("vae/configs/vae.yaml"))
    model = VanillaVAE(3, config["model_params"]["latent_dim"]).to("cuda")
    vae_exp = VAEXperiment(model, config)
    vae_exp.load_state_dict(
        torch.load("vae_logs/nico_dim/version_0/checkpoints/last.ckpt")[
            "state_dict"])
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = ResNetClassifier.load_from_checkpoint("/home/birk/Projects/robustSD/lightning_logs/version_0/checkpoints/epoch=199-step=1998200.ckpt", num_classes=num_classes, resnet_version=34).to("cuda")
    # classifier = SegmentationModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_1/checkpoints/epoch=48-step=2450.ckpt").to("cuda").eval()

    aconfig = {"device":"cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    # ds = robustSD(model, classifier, aconfig)

    # print("ood")
    ind_dataset_name = "nico"
    columns=["Feature Extractor", "Dataset", "Method", "Fold", "Sample Size", "P", "loss"]
    kn_data = []
    vanilla_data = []
    for sample_size in [10, 20, 50, 100, 200, 500, 1000, 10000]:
    # for sample_size in [10]:
        for context in os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train"):
            test_dataset = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context=context, seed=0)[1]
            kn_ps, ps, loss, test_loss = ds.compute_pvals_and_loss(ind, test_dataset, sample_size, ind_dataset_name= ind_dataset_name)
            kn_data.append(dict(zip(columns,
                            ["ResNet34", "NICO++", "KN", context, sample_size, kn_ps, np.mean(test_loss)])))
            vanilla_data.append(dict(zip(columns,
                            ["ResNet34", "NICO++", "Vanilla", context, sample_size, ps, np.mean(test_loss)])))
    kn_df = pd.DataFrame(kn_data, columns=columns)
    v_df = pd.DataFrame(vanilla_data, columns=columns)
    merged = pd.concat((kn_df, v_df))
    final = merged.explode(["P"])
    final.to_csv("data.csv")
    print(final.head(10))

