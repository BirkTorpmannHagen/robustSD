import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

from vae.vae_experiment import VAEXperiment
from segmentor.deeplab import SegmentationModel
from vae.models.vanilla_vae import VanillaVAE, ResNetVAE, CIFARVAE
import yaml
from torch.utils.data import RandomSampler
from classifier.resnetclassifier import ResNetClassifier
from ooddetectors import *
from torch.nn import functional as F
from classifier.cifarresnet import get_cifar, cifar10_pretrained_weight_urls, cifar100_pretrained_weight_urls

from njord.utils.loss import ComputeLoss
from njord.val import fetch_model
from njord.utils.dataloaders import LoadImagesAndLabels
# import segmentation_models_pytorch as smp

class BaseTestBed:
    def __init__(self, sample_size, num_workers=0):
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
    def __init__(self, sample_size, num_workers=20):
        super().__init__(sample_size, num_workers)
        self.noise_range = np.arange(0.05, 0.3, 0.05)


    def ind_loader(self):
        # return DataLoader(
        #     CIFAR10wNoise("../../Datasets/cifar10", train=True, transform=self.trans,noise_level=0), shuffle=False, num_workers=20)
        return DataLoader(
            self.ind, shuffle=True, num_workers=self.num_workers)

    def ind_val_loaders(self):
        loaders = {"ind": dict(
            [(sampler.__class__.__name__, DataLoader(self.ind_val, sampler=sampler, num_workers=self.num_workers)) for sampler in
             [ClassOrderSampler(self.ind_val, num_classes=self.num_classes),
              ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
              RandomSampler(self.ind_val)]])}
        return loaders

    def ood_loaders(self):


        oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=10), num_workers=self.num_workers),
                 DataLoader(test_dataset, sampler=ClusterSampler(test_dataset, self.rep_model, sample_size=self.sample_size), num_workers=self.num_workers),
                 DataLoader(test_dataset, sampler=RandomSampler(test_dataset), num_workers=self.num_workers)] for test_dataset in self.ood_sets]
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
        ind, ind_val, ood = build_njord_dataset()
        self.rep_model = ResNetVAE().to("cuda").eval()
        self.classifier = fetch_model("njord/runs/train/exp4/weights/best.pt")

        self.classifier.hyp = yaml.safe_load(open("/home/birk/Projects/robustSD/njord/data/hyps/hyp.scratch-low.yaml", "r"))
        self.loss = ComputeLoss(self.classifier)
        self.vae_exp = VAEXperiment(self.rep_model, yaml.safe_load(open("vae/configs/vae.yaml")))
        self.vae_exp.load_state_dict(
            torch.load("vae_logs/Njord/version_3/checkpoints/last.ckpt")[
                "state_dict"])
        self.ind, self.ind_val, self.ood = build_njord_dataset()
        self.collate_fn = LoadImagesAndLabels.collate_fn

    def loader(self, dataset, sampler):
        return DataLoader(dataset, sampler=sampler, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def compute_losses(self, loader):
        losses = torch.zeros(len(loader))
        for i, (x, targets, paths, shapes) in enumerate(loader):

            x = x.half()/255
            x = x.cuda()
            targets = targets.cuda()
            preds, train_out = self.classifier(x)
            _, loss = self.loss(train_out, targets)
            losses[i]=loss.mean().item()
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

    def __init__(self, sample_size, ood_noise=0):
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


        # self.rep_model = self.classifier
        # self.rep_model = ResNetVAE().to("cuda").eval()
        self.rep_model = self.classifier
        # self.vae_experiment = VAEXperiment(self.rep_model, yaml.safe_load(open("vae/configs/vae.yaml")))
        # self.vae_experiment.load_state_dict(torch.load("/home/birk/Projects/robustSD/vae_logs/nico_dim/version_7/checkpoints/last.ckpt")["state_dict"])


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
        return DataLoader(self.ind, shuffle=False, num_workers=20)


    def ood_loaders(self):

        test_datasets = [build_nico_dataset(1, "../../Datasets/NICO++", 0.2, self.trans, self.trans, context=context, seed=0)[1] for context in self.contexts]
        oods = [[DataLoader(test_dataset, sampler=ClassOrderSampler(test_dataset, num_classes=self.num_classes), num_workers=20),
                     DataLoader(test_dataset,
                                sampler=ClusterSampler(test_dataset, self.rep_model, sample_size=self.sample_size), num_workers=20),
                     DataLoader(test_dataset, sampler=RandomSampler(test_dataset), num_workers=20)] for test_dataset in test_datasets]

        dicted = [dict([(sampler, loader) for sampler, loader in zip(["ClassOrderSampler", "ClusterSampler", "RandomSampler"], ood)]) for ood in oods]
        double_dicted = dict([(context, dicted) for context, dicted in zip(self.contexts, dicted)])
        return double_dicted

    def ind_val_loaders(self):
        loaders =  {"ind": dict([(sampler.__class__.__name__, DataLoader(self.ind_val, sampler=sampler, num_workers=20)) for sampler in [ClassOrderSampler(self.ind_val, num_classes=self.num_classes),
                                                                              ClusterSampler(self.ind_val, self.rep_model, sample_size=self.sample_size),
                                                                              RandomSampler(self.ind_val)]])}
        return loaders


class CIFAR10TestBed(NoiseTestBed):
    def __init__(self, sample_size):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(), ])

        # classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to(
        #     "cuda").eval()
        # torch.save(classifier, "cifar10_model.pt")
        # for module in classifier.modules():
        #     print(module)
        self.classifier = get_cifar("resnet32", layers= [5]*3, model_urls=cifar10_pretrained_weight_urls,progress=True, pretrained=True).cuda().eval()

        self.rep_model = WrappedResnet(self.classifier)
        config = yaml.safe_load(open("vae/configs/vae.yaml"))
        self.vae = CIFARVAE().cuda().eval()
        vae_exp = VAEXperiment(self.vae, config)
        vae_exp.load_state_dict(
            torch.load("vae_logs/CIFAR10/version_1/checkpoints/epoch=95-step=300000.ckpt")[
                "state_dict"])
        self.num_classes = 10
        # self.ind_val = CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans, noise_level=0)
        self.ind, self.ind_val = torch.utils.data.random_split(CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans),[0.5, 0.5])
        self.oods = [CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans, noise_level=noise_val)
                                       for noise_val in self.noise_range]

        # return 0 #DEBUG

class CIFAR100TestBed(NoiseTestBed):
    def __init__(self, sample_size):
        super().__init__(sample_size)
        self.trans = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(), ])
        self.classifier = get_cifar("resnet32", layers=[5] * 3, model_urls=cifar100_pretrained_weight_urls,
                                    progress=True, pretrained=True, num_classes=100).cuda().eval()

        self.rep_model = WrappedResnet(self.classifier)
        config = yaml.safe_load(open("vae/configs/vae.yaml"))
        self.vae = CIFARVAE().cuda().eval()
        vae_exp = VAEXperiment(self.vae, config)
        vae_exp.load_state_dict(
            torch.load("vae_logs/CIFAR100/version_0/checkpoints/epoch=89-step=281250.ckpt")[
                "state_dict"])
        self.num_classes = 100
        # self.ind_val = CIFAR10wNoise("../../Datasets/cifar10", train=False, transform=self.trans, noise_level=0)
        self.ind, self.ind_val = torch.utils.data.random_split(
            CIFAR100wNoise("../../Datasets/cifar100", train=False, transform=self.trans, download=True), [0.5, 0.5])
        self.oods = [CIFAR100wNoise("../../Datasets/cifar100", train=False, transform=self.trans, noise_level=noise_val)
                                       for noise_val in self.noise_range]

class ImagenetteTestBed(NoiseTestBed):
    def __init__(self, sample_size: object) -> object:
        """

        :type sample_size: object
        """
        super().__init__(sample_size)

        self.trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(), ])

        # classifier = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).to(
        #     "cuda").eval()
        # torch.save(classifier, "cifar10_model.pt")
        # for module in classifier.modules():
        #     print(module)
        # self.classifier = ResNetClassifier(num_classes=10, resnet_version=101).cuda().eval()
        self.classifier = ResNetClassifier.load_from_checkpoint("Imagenette_logs/checkpoints/epoch=82-step=24568.ckpt", resnet_version=101,num_classes=10)
        self.classifier.eval().to("cuda")
        self.rep_model = self.classifier

        config = yaml.safe_load(open("vae/configs/vae.yaml"))
        self.vae = ResNetVAE().cuda().eval()
        vae_exp = VAEXperiment(self.vae, config)
        vae_exp.load_state_dict(
            torch.load("vae_logs/Imagenette/version_2/checkpoints/last.ckpt")[
                "state_dict"])

        self.num_classes = 10
        self.ind, self.ind_val = build_imagenette_dataset("../../Datasets/imagenette2", self.trans,self.trans)
        self.ood_sets =  [ImagenettewNoise("../../Datasets/imagenette2", train="val", transform=self.trans, noise_level=noise_val)
                                       for noise_val in self.noise_range]

class PolypTestBed(BaseTestBed):
    def __init__(self, sample_size):
        super().__init__(sample_size)
        # self.ind, self.val, self.ood = build_polyp_dataset("../../Datasets/Polyp")
        # self.rep_model = SegmentationModel()
        # dict = torch.load("segmentation_logs/lightning_logs/version_11/checkpoints/epoch=142-step=23023.ckpt")
        # self.rep_model.load_state_dict(dict)
        self.trans = [alb.Compose([
            i,
            alb.Resize(512, 512)]) for i in [alb.HorizontalFlip(p=0), alb.HorizontalFlip(always_apply=True),
                                             alb.VerticalFlip(always_apply=True), alb.RandomRotate90(always_apply=True),
                                             ]]


        inds = []
        vals = []
        oods = []
        for i in self.trans:

            cvc_train_set, cvc_val_set = build_polyp_dataset("../../Datasets/Polyps/CVC-ClinicDB", trans=i,fold="CVC", seed=0)
            kvasir_train_set, kvasir_val_set = build_polyp_dataset("../../Datasets/Polyps/HyperKvasir",trans=i, fold="Kvasir", seed=0)
            etis_train, etis_val = build_polyp_dataset("../../Datasets/Polyps/ETIS-LaribPolypDB", trans=i, fold="Etis", seed=0)
            inds.append(kvasir_train_set)
            inds.append(kvasir_val_set)
            vals.append(cvc_val_set)
            vals.append(cvc_train_set)
            oods.append(etis_train)
            oods.append(etis_val)


        self.ind = ConcatDataset(inds)
        self.ind_val = ConcatDataset(vals)
        self.ood = ConcatDataset(oods)

        params = {
            "LR": 0.00005,
            "weight_decay": 0.0,
            "scheduler_gamma": 0.95,
            "kld_weight": 0.00025,
            "manual_seed": 1265

        }
        self.vae = ResNetVAE().to("cuda").eval()
        vae_exp = VAEXperiment(self.vae, params)
        vae_exp.load_state_dict(
            torch.load("vae_logs/KvasirSegmentationDataset/version_0/checkpoints/epoch=339-step=34000.ckpt")[
                "state_dict"])


        self.classifier = SegmentationModel.load_from_checkpoint(
            "segmentation_logs/lightning_logs/version_11/checkpoints/epoch=142-step=23023.ckpt").to("cuda")
        self.classifier.eval()
        self.rep_model = self.vae

    def ind_loader(self):
        return DataLoader(self.ind)

    def ood_loaders(self):
        samplers = [ClusterSampler(self.ood, self.rep_model, sample_size=self.sample_size),
                                  SequentialSampler(self.ood), RandomSampler(self.ood)]
        loaders =  {"ood": dict([[sampler.__class__.__name__,  DataLoader(self.ood, sampler=sampler)] for sampler in
                                 samplers])}
        return loaders


    def ind_val_loaders(self):
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

