import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from yellowbrick.features import PCA
from vae.vae_experiment import VAEXperiment
from vae.models.vanilla_vae import VanillaVAE, ResNetVAE
import yaml
import torch
from torch.utils.data import DataLoader
from yellowbrick.features.manifold import Manifold
from sklearn.manifold import SpectralEmbedding, Isomap
from scipy.stats import ks_2samp
from bias_samplers import ClusterSampler
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


    def compute_pvals_and_loss(self, ind_dataset, ood_dataset, ood_sampler, sample_size, ind_dataset_name, ood_dataset_name, k=5, plot=False):
        sample_size = min(sample_size, len(ood_dataset))
        fname_encodings = f"robustSD_{ind_dataset_name}_enodings_{type(self.rep_model).__name__}.pkl"
        fname_losses = f"robustSD_{ind_dataset_name}_losses_{type(self.rep_model).__name__}.pkl"
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
                # if ood_losses[i]<0.01:
                #     print(torch.argmax(self.classifier(x.to(self.config["device"])).sigmoid()))
                #     print(y)
                #     print("huh")
                #     plt.imshow(x[0].cpu().numpy().T)
                #     plt.show()
        cols = ["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
        dataframe = []

        for start, stop in list(zip(range(0, len(ood_dataset), sample_size), range(sample_size, len(ood_dataset)+sample_size, sample_size)))[:-1]: #perform tests
            sample_idx = range(start,stop)
            ood_samples = ood_latents[sample_idx]
            vanilla_pval = min(np.min([ks_2samp(ind_latents[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)]), 1)
            k_nearest_idx = np.concatenate([np.argpartition(np.sum((np.expand_dims(i, 0) - ind_latents) ** 2, axis=-1), k)[:k] for i in ood_samples])
            k_nearest_ind = ind_latents[k_nearest_idx]
            # k_nearest_idx = [np.argmin(np.sum((np.expand_dims(i, 0) - ind_latents) ** 2, axis=-1)) for i in ood_samples]
            # k_nearest_ind = ind_latents[k_nearest_idx]
            # viz = PCA()
            # viz.fit_transform(X=np.concatenate((ind_latents, ood_samples, k_nearest_ind)), y=[0]*len(ind_latents)+[1]*len(ood_samples)+[2]*len(k_nearest_ind))
            # viz.show()
            kn_pval = min(np.min([ks_2samp(k_nearest_ind[:,i], ood_samples[:, i])[-1] for i in range(self.rep_model.latent_dim)]), 1)
            dataframe.append(dict(zip(cols, [ind_dataset_name, ood_dataset_name, type(self.rep_model).__name__, sample_size, vanilla_pval, kn_pval, np.mean(ood_losses[sample_idx])])))
        final = pd.DataFrame(data=dataframe, columns=cols)
        print(final.head())
        return final
    def bootstrap_severity_estimation(self):
        pass

    def eval_synthetic(self, ind_dataset, ind_val, trans_fn, sampler, sample_size, ind_dataset_name, ood_dataset_name, plot=False):
        dataset = transform_dataset(ind_val, trans_fn)
        return self.compute_pvals_and_loss(ind_dataset, dataset, ood_sampler=sampler,sample_size=sample_size, ind_dataset_name=ind_dataset_name, ood_dataset_name=ood_dataset_name, plot=plot)


def eval_nico():
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))

    ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
    # ind, ind_val = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    config = yaml.safe_load(open("vae/configs/vae.yaml"))
    model = ResNetVAE().to("cuda").eval()
    vae_exp = VAEXperiment(model, config)
    vae_exp.load_state_dict(
        torch.load("/home/birk/Projects/robustSD/vae_logs/nico_dim/version_40/checkpoints/epoch=8-step=22482.ckpt")[
            "state_dict"])
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = ResNetClassifier.load_from_checkpoint("/home/birk/Projects/robustSD/lightning_logs/version_0/checkpoints/epoch=199-step=1998200.ckpt", num_classes=num_classes, resnet_version=34).to("cuda")
    # classifier = SegmentationModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_1/checkpoints/epoch=48-step=2450.ckpt").to("cuda").eval()

    aconfig = {"device":"cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    # ds = robustSD(model, classifier, aconfig)

    # print("ood")
    ind_dataset_name = "nico"
    columns=["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
    merged = []
    for context in os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train"):
        test_dataset = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context=context, seed=0)[1]
        for sample_size in [10, 20, 50, 100, 200, 500, 1000, 10000]:
            data = ds.compute_pvals_and_loss(ind, test_dataset, ood_sampler=ClusterSampler(test_dataset, classifier, sample_size=sample_size),
                                             sample_size=sample_size, ind_dataset_name="nico_dim", ood_dataset_name=f"nico_{context}")
            merged.append(data)
    final = pd.concat(merged)
    final.to_csv(f"lp_nico_datak6  .csv")
    print(final.head(10))

def nico_correlation():
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(), ])
    contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
    # datasets = dict(zip(contexts, [build_dataset(1, "datasets/NICO++", 0, trans, trans, context=i, seed=0) for i in contexts]))

    ind, ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, trans, trans, context="dim", seed=0)
    # ind, ind_val = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir", "Kvasir", 0)
    # config = yaml.safe_load(open("vae/configs/vae.yaml"))
    # model = ResNetVAE().to("cuda").eval()
    # vae_exp = VAEXperiment(model, config)
    # vae_exp.load_state_dict(
    #     torch.load("vae_logs/nico_dim/version_40/checkpoints/last.ckpt")[
    #         "state_dict"])
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = ResNetClassifier.load_from_checkpoint("/home/birk/Projects/robustSD/lightning_logs/version_0/checkpoints/epoch=109-step=1236510.ckpt", num_classes=num_classes, resnet_version=34).to("cuda")
    # classifier = SegmentationModel.load_from_checkpoint("segmentation_logs/lightning_logs/version_1/checkpoints/epoch=48-step=2450.ckpt").to("cuda").eval()

    aconfig = {"device":"cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    # ds = robustSD(model, classifier, aconfig)

    # print("ood")
    ind_dataset_name = "nico"
    columns=["ind_dataset", "ood_dataset", "rep_model", "sample_size", "vanilla_p", "kn_p", "loss"]
    merged = []
    for noise_val in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.50, 0.75]:
        for sample_size in [10, 20, 50, 100, 200, 500, 1000, 10000]:
            data = ds.eval_synthetic(ind, ind_val,lambda x: x+torch.randn_like(x)*noise_val, sampler=ClusterSampler(ind_val, classifier, sample_size=sample_size),
                                             sample_size=sample_size, ind_dataset_name="nico_dim", ood_dataset_name=f"nico_{noise_val}")
            merged.append(data)
    final = pd.concat(merged)
    final.to_csv(f"lp_data_nico_noise.csv")
    print(final.head(10))



if __name__ == '__main__':
    # generate_plot(create_datasets_by_fold(), ["ind", "test_val"])
    nico_correlation()

