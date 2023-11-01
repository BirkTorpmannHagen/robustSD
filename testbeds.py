import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset

from vae.vae_experiment import VAEXperiment
from segmentor.deeplab import SegmentationModel
from vae.models.vanilla_vae import VanillaVAE, ResNetVAE, CIFARVAE
import yaml
from torch.utils.data import RandomSampler
from classifier.resnetclassifier import ResNetClassifier
from ooddetectors import *
import itertools

from torch.nn import functional as F
from classifier.cifarresnet import get_cifar, cifar10_pretrained_weight_urls, cifar100_pretrained_weight_urls
from domain_datasets import *
from njord.utils.loss import ComputeLoss
from njord.val import fetch_model
from njord.utils.dataloaders import LoadImagesAndLabels
# import segmentation_models_pytorch as smp
DEFAULT_PARAMS = {
    "LR": 0.00005,
    "weight_decay": 0.0,
    "scheduler_gamma": 0.95,
    "kld_weight": 0.00025,
    "manual_seed": 1265

}
class BaseTestBed:
    def __init__(self, sample_size, num_workers=5):
        self.sample_size = sample_size
        self.num_workers=num_workers


    def compute_losses(self, loaders):
        pass

    def ind_loader(self):
        pass

    def ood_loaders(self):
        pass

    def ind_val_loaders(self):
        pass

class NoiseTestBed(BaseTestBed):
    def __init__(self, sample_size, num_workers=5, mode="normal"):
        super().__init__(sample_size, num_workers)
        self.num_workers=num_workers
        self.noise_range = np.arange(0.0, 0.35, 0.05)[1:]
        self.mode=mode


    def ind_loader(self):
        # return DataLoader(
        #     CIFAR10wNoise("../../Datasets/cifar10", train=True, transform=self.trans,noise_level=0), shuffle=False, num_workers=5)
        return DataLoader(
            self.ind, shuffle=True, num_workers=self.num_workers)

    def ind_val_loaders(self):
        if self.mode=="severity":
            samplers = [ClusterSamplerWithSeverity(self.ind_val, self.rep_model, sample_size=self.sample_size, bias_severity=i) for i in np.linspace(0,1,11)]
            loaders = {"ind": dict(
                [(str(sampler), DataLoader(self.ind_val, sampler=sampler, num_workers=self.num_workers))
                 for sampler in
                 samplers])}
        else:
            loaders = {"ind": dict(
                [(sampler.__class__.__name__, DataLoader(self.ind_val, sampler=sampler, num_workers=self.num_workers)) for sampler in
                 [ClassOrderSampler(self.ind_val, num_classes=self.num_classes),
                  ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
                  RandomSampler(self.ind_val)]])}
        return loaders

    def ood_loaders(self):
        if self.mode=="severity":
            #a sampler for each ood_set and severity

            combinations =  itertools.product(self.oods, np.linspace(0,1,11))
            oods = {}
            for i, ood_set in enumerate(self.oods):
                oods_by_bias = {}
                for severity in np.linspace(0,1,11):
                    oods_by_bias[f"ClusterBias:{severity}"] = DataLoader(ood_set,
                                sampler=ClusterSamplerWithSeverity(ood_set, self.rep_model,
                                    sample_size=self.sample_size, bias_severity=severity),
                                num_workers=self.num_workers)
                oods[f"noise_{self.noise_range[i]}"] = oods_by_bias
            return oods
        else:
            oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=self.num_classes), num_workers=self.num_workers),
                     DataLoader(test_dataset, sampler=ClusterSampler(test_dataset, self.rep_model, sample_size=self.sample_size), num_workers=self.num_workers),
                     DataLoader(test_dataset, sampler=RandomSampler(test_dataset), num_workers=self.num_workers)] for test_dataset in self.oods]
            dicted = [dict([(sampler, loader) for sampler, loader in zip(["ClassOrderSampler", "ClusterSampler", "RandomSampler"], ood)]) for ood in oods]
            double_dicted = dict(zip(["noise_{}".format(noise_val) for noise_val in self.noise_range], dicted))
            return double_dicted
    def compute_losses(self, loader):
        losses = torch.zeros(len(loader) ).to("cuda")
        # accs = torch.zeros(len(loader) ).to("cuda")
        # from torchmetrics import Accuracy
        # acc = Accuracy("multiclass", num_classes=self.num_classes).cuda()
        criterion = nn.CrossEntropyLoss()
        for i, data in tqdm(enumerate(loader), total=len(loader)):

            x = data[0].to("cuda")
            y = data[1].to("cuda")
            yhat = self.classifier(x)
            # accs[i] =acc(yhat, y).item()
            losses[i]=criterion(yhat, y).item()
        return losses.cpu().numpy()

class NjordTestBed(BaseTestBed):
    def __init__(self, sample_size):
        super().__init__(sample_size)
        self.rep_model = self.vae = VanillaVAE(3, 512).to("cuda").eval()
        self.classifier = fetch_model("njord/runs/train/exp4/weights/best.pt")

        self.classifier.hyp = yaml.safe_load(open("/home/birk/Projects/robustSD/njord/data/hyps/hyp.scratch-low.yaml", "r"))
        self.loss = ComputeLoss(self.classifier)
        self.vae_exp = VAEXperiment(self.rep_model, DEFAULT_PARAMS)
        self.vae_exp.load_state_dict(
            torch.load("vae_logs/Njord/version_1/checkpoints/last.ckpt")[
                "state_dict"])
        self.ind, self.ind_val, self.ood = build_njord_datasets()
        self.collate_fn = LoadImagesAndLabels.collate_fn

    def loader(self, dataset, sampler):
        return DataLoader(dataset, sampler=sampler, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def split_datasets(self):
        val = DataLoader(self.ind_val, collate_fn = self.collate_fn)
        ood = DataLoader(self.ood, collate_fn = self.collate_fn)

        val_losses = torch.zeros(len(val))
        val_data = []
        print("getting val losses")

        for i, (x, targets, paths, shapes) in tqdm(enumerate(val), total=len(val)):
            x = x.half()/255
            x = x.cuda()
            targets = targets.cuda()
            preds, train_out = self.classifier(x)
            loss, _ = self.loss(train_out, targets)
            val_losses[i]=loss.item()
            val_data.append({"fold": "val", "path": paths, "loss":loss.item()})
        df_val = pd.DataFrame(val_data)
        df_val.to_csv("njord_losses_val.csv")

        ood_losses = torch.zeros(len(ood))
        ood_data = []
        print("getting ood losses")
        for i, (x, targets, paths, shapes) in tqdm(enumerate(ood), total=len(ood)):
            x = x.half() / 255
            x = x.cuda()
            targets = targets.cuda()
            preds, train_out = self.classifier(x)
            loss, _ = self.loss(train_out, targets)
            ood_losses[i] = loss.item()
            ood_data.append({"fold": "ood", "path": paths, "loss": loss.item()})
        df_ood = pd.DataFrame(ood_data)
        df_ood.to_csv("njord_losses_ood.csv")

        merged = pd.concat((df_val, df_ood))
        merged.to_csv("merged_njord_loses.csv")
        import seaborn as sns
        sns.kdeplot(data=merged, x="loss", hue="fold")
        plt.savefig("figures/njord_losskde.png")
        plt.show()

        input()

    def compute_losses(self, loader):
        losses = np.zeros(len(loader))
        for i, (x, targets, paths, shapes) in enumerate(loader):
            x = x.half()/255
            x = x.cuda()
            targets = targets.cuda()
            preds, train_out = self.classifier(x)
            loss, _ = self.loss(train_out, targets)
            losses[i]=loss.item()
        return losses

    def ind_loader(self):
        return self.loader(self.ind, sampler=RandomSampler(self.ind))

    def ood_loaders(self):
        samplers = [ClusterSampler(self.ood, self.rep_model, sample_size=self.sample_size),
                                  SequentialSampler(self.ood), RandomSampler(self.ood)]
        loaders =  {"ood": dict([[sampler.__class__.__name__,  self.loader(self.ood, sampler=sampler)] for sampler in
                                 samplers])}
        return loaders


    def ind_val_loaders(self):
        samplers = [ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
                                  SequentialSampler(self.ind_val), RandomSampler(self.ind_val)]

        loaders =  {"ind": dict([ [sampler.__class__.__name__,  self.loader(self.ind_val, sampler=sampler)] for sampler in
                                 samplers])}
        return  loaders


class NicoTestBed(BaseTestBed):

    def __init__(self, sample_size, rep_model="vae", ood_noise=False, mode="severity"):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
                                                 transforms.Resize((512, 512)),
                                                 transforms.ToTensor(), ])
        self.num_classes = num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))

        self.ind, self.ind_val = build_nico_dataset(1, "../../Datasets/NICO++", 0.2, self.trans, self.trans, context="dim", seed=0)
        self.num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
        self.contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
        print(self.contexts)
        self.contexts.remove("dim")
            # self.classifier = ResNetClassifier.load_from_checkpoint(
            #     "lightning_logs/version_0/checkpoints/epoch=199-step=1998200.ckpt", num_classes=num_classes,
            #     resnet_version=34).to("cuda").eval()
        self.classifier = ResNetClassifier.load_from_checkpoint(
           "NICODataset_logs/checkpoints/epoch=279-step=175000.ckpt", num_classes=num_classes,
            resnet_version=101).to("cuda").eval()
        self.vae = VanillaVAE(3, 512).to("cuda").eval()

        if rep_model=="vae":
            self.rep_model = self.vae
            print("USING VAE!!")
        else:
            self.rep_model=self.classifier
        self.vae_experiment = VAEXperiment(self.rep_model, DEFAULT_PARAMS)
        self.vae_experiment.load_state_dict(torch.load("vae_logs/NICODataset/version_0/checkpoints/epoch=104-step=131040.ckpt")["state_dict"])
        self.noise_range = np.arange(0.05, 0.3, 0.05)
        self.ood_noise = ood_noise

    def compute_losses(self, loader):
        losses = torch.zeros(len(loader)).to("cuda")
        print("computing losses")
        for i, (x, y, _) in tqdm(enumerate(loader), total=len(loader)):
            x = x.to("cuda")
            y = y.to("cuda")
            yhat = self.classifier(x)
            losses[i]=F.cross_entropy(yhat, y).item()
        return losses.cpu().numpy()

    def ind_loader(self):
        return DataLoader(self.ind, shuffle=False, num_workers=5)


    def ood_loaders(self):
        if self.ood_noise:
            ood_sets = [NoisyDataset(self.ind_val, noise) for noise in self.noise_range]
            oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=self.num_classes),
                                num_workers=self.num_workers),
                     DataLoader(test_dataset,
                                sampler=ClusterSampler(test_dataset, self.rep_model, sample_size=self.sample_size),
                                num_workers=self.num_workers),
                     DataLoader(test_dataset, sampler=RandomSampler(test_dataset), num_workers=self.num_workers)] for
                    test_dataset in ood_sets]
            dicted = [dict([(sampler, loader) for sampler, loader in
                            zip(["ClassOrderSampler", "ClusterSampler", "RandomSampler"], ood)]) for ood in oods]
            double_dicted = dict(zip(["noise_{}".format(noise_val) for noise_val in self.noise_range], dicted))
            return double_dicted
        else:
            test_datasets = [build_nico_dataset(1, "../../Datasets/NICO++", 0.2, self.trans, self.trans, context=context, seed=0)[1] for context in self.contexts]
            oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=self.num_classes), num_workers=5),
                         DataLoader(test_dataset,
                                    sampler=ClusterSampler(test_dataset, self.rep_model, sample_size=self.sample_size), num_workers=5),
                         DataLoader(test_dataset, sampler=RandomSampler(test_dataset), num_workers=5)] for test_dataset in test_datasets]

            dicted = [dict([(sampler, loader) for sampler, loader in zip(["ClassOrderSampler", "ClusterSampler", "RandomSampler"], ood)]) for ood in oods]
            double_dicted = dict([(context, dicted) for context, dicted in zip(self.contexts, dicted)])
            return double_dicted

    def ind_val_loaders(self):
        loaders =  {"ind": dict([(sampler.__class__.__name__, DataLoader(self.ind_val, sampler=sampler, num_workers=5)) for sampler in [ClassOrderSampler(self.ind_val, num_classes=self.num_classes),
                                                                              ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
                                                                              RandomSampler(self.ind_val)]])}
        return loaders


class CIFAR10TestBed(NoiseTestBed):
    def __init__(self, sample_size, rep_model,mode):
        super().__init__(sample_size, mode=mode)
        self.trans = transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(), ])

        # classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to(
        #     "cuda").eval()
        # torch.save(classifier, "cifar10_model.pt")
        # for module in classifier.modules():
        #     print(module)
        self.classifier = get_cifar("resnet32", layers= [5]*3, model_urls=cifar10_pretrained_weight_urls,progress=True, pretrained=True).cuda().eval()

        self.classifier = WrappedResnet(self.classifier)
        self.vae = CIFARVAE().cuda().eval()
        vae_exp = VAEXperiment(self.vae, DEFAULT_PARAMS)
        vae_exp.load_state_dict(
            torch.load("vae_logs/CIFAR10/version_35/checkpoints/epoch=121-step=762500.ckpt")[
                "state_dict"])
        self.num_classes = 10
        # self.ind_val = CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans, noise_level=0)
        self.ind, self.ind_val = torch.utils.data.random_split(CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans),[0.5, 0.5])
        self.oods = [CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans, noise_level=noise_val)
                                       for noise_val in self.noise_range]
        if rep_model=="vae":
            self.rep_model = self.vae
        else:
            self.rep_model=self.classifier

        # return 0 #DEBUG

class CIFAR100TestBed(NoiseTestBed):
    def __init__(self, sample_size, rep_model, mode="normal"):
        super().__init__(sample_size, mode=mode)
        self.trans = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(), ])
        self.classifier = get_cifar("resnet32", layers=[5] * 3, model_urls=cifar100_pretrained_weight_urls,
                                    progress=True, pretrained=True, num_classes=100).cuda().eval()

        self.classifier= WrappedResnet(self.classifier)

        self.num_classes = 100
        # self.ind_val = CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans, noise_level=0)
        self.ind, self.ind_val = torch.utils.data.random_split(
            CIFAR100wNoise("../../Datasets/cifar100", train=False, transform=self.trans, download=True), [0.5, 0.5])
        self.oods = [CIFAR100wNoise("../../Datasets/cifar100", train=False, transform=self.trans, noise_level=noise_val)
                                       for noise_val in self.noise_range]
        if rep_model=="vae":
            config = yaml.safe_load(open("vae/configs/vae.yaml"))
            self.vae = CIFARVAE().cuda().eval()
            vae_exp = VAEXperiment(self.vae, config)
            vae_exp.load_state_dict(
                torch.load("vae_logs/CIFAR100/version_0/checkpoints/epoch=89-step=281250.ckpt")[
                    "state_dict"])
            self.rep_model = self.vae
        else:
            self.rep_model=self.classifier

class ImagenetteTestBed(NoiseTestBed):
    def __init__(self, sample_size, rep_model, mode="normal"):
        """

        :type sample_size: object
        """
        super().__init__(sample_size, num_workers=5, mode=mode)

        self.trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(), ])

        # classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to(
        #     "cuda").eval()
        # torch.save(classifier, "cifar10_model.pt")
        # for module in classifier.modules():
        #     print(module)
        # self.classifier = ResNetClassifier(num_classes=10, resnet_version=101).cuda().eval()
        self.classifier = ResNetClassifier.load_from_checkpoint("Imagenette_logs/checkpoints/epoch=82-step=24568.ckpt", resnet_version=101,num_classes=10)
        self.classifier.eval().to("cuda")

        config = yaml.safe_load(open("vae/configs/vae.yaml"))
        self.vae = VanillaVAE(3, 512).cuda().eval()
        vae_exp = VAEXperiment(self.vae, config)
        # vae_exp.load_state_dict(
        #     torch.load("vae_logs/Imagenette/version_0/checkpoints/epoch=106-step=126581.ckpt")[
        #         "state_dict"])

        self.num_classes = 10
        self.ind, self.ind_val = build_imagenette_dataset("../../Datasets/imagenette2", self.trans,self.trans)
        self.ood_sets =  [ImagenettewNoise("../../Datasets/imagenette2", train="val", transform=self.trans, noise_level=noise_val)
                                       for noise_val in self.noise_range]
        if rep_model=="vae":
            self.rep_model = self.vae
        else:
            self.rep_model=self.classifier

class PolypTestBed(BaseTestBed):
    def __init__(self, sample_size, rep_model, mode="normal"):
        super().__init__(sample_size)
        self.ind, self.ind_val, self.ood = build_polyp_dataset("../../Datasets/Polyps")
        self.noise_range = np.arange(0.05, 0.3, 0.05)
        #vae
        self.vae = VanillaVAE(in_channels=3, latent_dim=512).to("cuda").eval()
        vae_exp = VAEXperiment(self.vae, DEFAULT_PARAMS)
        vae_exp.load_state_dict(
            torch.load("vae_logs/Polyp/version_0/checkpoints/epoch=116-step=75348.ckpt")[
                "state_dict"])

        #segmodel
        self.classifier = SegmentationModel.load_from_checkpoint(
            "segmentation_logs/lightning_logs/version_12/checkpoints/epoch=199-step=64600.ckpt").to("cuda")
        self.classifier.eval()

        #assign rep model
        if rep_model == "vae":
            self.rep_model = self.vae
        else:
            self.rep_model = self.classifier

        self.mode = mode

    def ind_loader(self):
        return DataLoader(self.ind)

    def ood_loaders(self):
        if self.mode=="noise":
            ood_sets = [NoisyDataset(self.ind_val, noise) for noise in self.noise_range]
            oods = [[DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                num_workers=self.num_workers),
                     DataLoader(test_dataset,
                                sampler=ClusterSampler(test_dataset, self.rep_model, sample_size=self.sample_size),
                                num_workers=self.num_workers),
                     DataLoader(test_dataset, sampler=RandomSampler(test_dataset), num_workers=self.num_workers)] for
                    test_dataset in ood_sets]
            dicted = [dict([(sampler, loader) for sampler, loader in
                            zip(["ClassOrderSampler", "ClusterSampler", "RandomSampler"], ood)]) for ood in oods]
            double_dicted = dict(zip(["noise_{}".format(noise_val) for noise_val in self.noise_range], dicted))
            return double_dicted
        elif self.mode=="severity":
            samplers = [ClusterSamplerWithSeverity(self.ood, self.rep_model, sample_size=self.sample_size,
                                                   bias_severity=severity) for severity in np.linspace(0,1,11)]
            loaders =  {"ood": dict([[sampler.__class__.__name__,  DataLoader(self.ood, sampler=sampler)] for sampler in
                                     samplers])}
            return loaders
        else:
            samplers = [ClusterSampler(self.ood, self.rep_model, sample_size=self.sample_size),
                                      SequentialSampler(self.ood), RandomSampler(self.ood)]
            loaders =  {"ood": dict([[sampler.__class__.__name__,  DataLoader(self.ood, sampler=sampler)] for sampler in
                                     samplers])}
            return loaders


    def ind_val_loaders(self):
        if self.mode=="severity":
            samplers = [ClusterSamplerWithSeverity(self.ood, self.rep_model, sample_size=self.sample_size,
                                                   bias_severity=severity) for severity in np.linspace(0,1,11)]
            loaders =  {"ood": dict([[sampler.__class__.__name__,  DataLoader(self.ood, sampler=sampler)] for sampler in
                                     samplers])}
            return loaders
        else:
            samplers = [ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
                                      SequentialSampler(self.ind_val), RandomSampler(self.ind_val)]

            loaders =  {"ind": dict([ [sampler.__class__.__name__,  DataLoader(self.ind_val, sampler=sampler)] for sampler in
                                     samplers])}
            return loaders

    def compute_losses(self, loader):
        losses = np.zeros(len(loader))
        print("computing losses")
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            x = data[0].to("cuda")
            y = data[1].to("cuda")
            losses[i]=self.classifier.compute_loss(x,y).item()
        return losses

