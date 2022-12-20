import os

import matplotlib.pyplot as plt
import numpy as np

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
from domain_datasets import build_nico_dataset
from classifier.resnetclassifier import ResNetClassifier
from scipy.stats import wasserstein_distance
class robustSD:
    def __init__(self, rep_model, classifier, config):
        self.rep_model = rep_model
        self.classifier = classifier
        self.config = config
        self.pca = PCA(2)


    def compute_pvals_and_loss(self, ind_dataset, ood_dataset, sample_size, plot=False):
        sample_size = min(sample_size, len(ood_dataset))
        ind_dataset_name = ""
        ood_dataset_name = ""
        fname_encodings = f"robustSD_{ind_dataset_name}_enodings"
        fname_losses = f"robustSD_{ind_dataset_name}_losses"

        try:
            latents = pkl.load(open(fname_encodings, "rb"))
            losses = pkl.load(open(fname_losses, "rb"))
        except FileNotFoundError:
            latents = np.zeros((len(ind_dataset), self.rep_model.latent_dim))
            losses = np.zeros(len(ind_dataset))

            for i, (x, y, _) in tqdm(enumerate(DataLoader(ind_dataset)),total=len(ind_dataset)):
                with torch.no_grad():
                    latents[i] = self.rep_model.encode(x.to(self.config["device"])).cpu().numpy()
                    losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),y.to(self.config["device"])).cpu().numpy()
            pkl.dump(latents, open(fname_encodings, "wb"))
            pkl.dump(losses, open(fname_losses, "wb"))

        ood_latents = np.zeros((len(ood_dataset), self.rep_model.latent_dim))
        ood_losses = np.zeros(len(ood_dataset))
        for i, (x, y, _) in tqdm(enumerate(DataLoader(ood_dataset)), total=len(ood_dataset)):
            with torch.no_grad():
                ood_latents[i] = self.rep_model.encode(x.to(self.config["device"])).cpu().numpy()
                # TODO there exists duplicate images in separate contexts. fix?
                # if ood_latents[i].tolist() in latents.tolist():
                #
                #     plt.imshow(x[0].T.cpu().numpy())
                #     print(y)
                #     plt.show()
                ood_losses[i] = self.classifier.compute_loss(x.to(self.config["device"]),
                                                         y.to(self.config["device"])).cpu().numpy()

        # self.pca.fit(np.vstack((latents, ood_latents)))
        # latents = self.pca.transform(latents)
        # ood_latents = self.pca.transform(ood_latents)

        k_n_indx = [np.argmin(np.sum((np.expand_dims(i, 0) - latents) ** 2, axis=-1)) for i in ood_latents]
        k_n_dists = [np.min(np.sum((np.expand_dims(i, 0) - latents) ** 2, axis=-1)) for i in ood_latents]
        k_nearest = latents[k_n_indx]
        # plt.scatter(latents[:, 0], latents[:, 1], label="ind")
        if plot:
            plt.scatter(ood_latents[:, 0], ood_latents[:, 1], label="ood")
            plt.scatter(k_nearest[:, 0], k_nearest[:, 1], label="kn")
            plt.legend()
            plt.show()

        p_vals_kn = []
        # p_vals_basic = []
        print(wasserstein_distance(ood_latents[:,0], k_nearest[:,0]))
        print(wasserstein_distance(ood_latents[:,1], k_nearest[:,1]))

        for j in range(25):
            sample_idx = np.random.choice(range(len(ood_latents)), sample_size)
            subsample_ind = k_nearest[sample_idx,:]
            subsample_ood = ood_latents[sample_idx,:]
            if plot:
                plt.scatter(subsample_ood[:,0],subsample_ood[:,1],label="ood")
                plt.scatter(subsample_ind[:, 0], subsample_ind[:, 1], label="ind")
                plt.legend()
                plt.title(str(j))
                plt.show()
            p_vals_kn.append(
                min(np.min([ks_2samp(subsample_ind[:,i], subsample_ood[:, i])[-1] for i in range(512)]) * 512, 1)
            )
            # p_vals_basic.append(
            #     min(np.min([ks_2samp(latents[:, i], ood_latents[sample_idx, i]) for i in range(2)]) * 2, 1)
            # )
        print(p_vals_kn)
        return np.array(p_vals_kn), (losses, ood_losses)


    def bootstrap_severity_estimation(self):
        pass

    def eval_given_transforms(self, dataset, transforms):
        for transform in transforms:
            pass




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

    config = yaml.safe_load(open("vae/configs/vae.yaml"))
    model = VanillaVAE(3, config["model_params"]["latent_dim"]).to("cuda")
    vae_exp = VAEXperiment(model, config)
    vae_exp.load_state_dict(
        torch.load("VAEs/nico_dim/version_0/checkpoints/last.ckpt")[
            "state_dict"])
    num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
    classifier = ResNetClassifier.load_from_checkpoint("lightning_logs/version_2/checkpoints/epoch=55-step=559496.ckpt", num_classes=num_classes, resnet_version=34).to("cuda")
    aconfig = {"device":"cuda"}
    ds = robustSD(classifier, classifier, aconfig)
    print(set(ind_val.image_path_list).intersection(ind.image_path_list))
    # print("ood")
    ds.compute_pvals_and_loss(ind, ood_1, 500)
    ds.compute_pvals_and_loss(ind, ood_2, 500)
    ds.compute_pvals_and_loss(ind, ood_3, 500)
    print("ind")
    ds.compute_pvals_and_loss(ind, ind_val, 500)


