import matplotlib.pyplot as plt
import torch
from torch.utils.data import Sampler, DataLoader
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from yellowbrick.features import PCA

class ClassOrderSampler(Sampler):
    """
    Sampler that splits the data_source into classes, returns indexes in order of class
    Induces selection bias via label shift
    """
    def __init__(self, data_source, num_classes):
        super(ClassOrderSampler, self).__init__(data_source)
        self.data_source = data_source
        self.indices = [[] for i in range(num_classes)]

        #initial pass to sort the indices by class
        for i, data in enumerate(data_source):
            self.indices[data[1]].append(i)


    def __iter__(self):
        return iter(sum(self.indices, []))

    def __len__(self):
        return len(self.data_source)


class SequentialSampler(Sampler):
    """
    Samples sequentially from the dataset (assuming the dataloader fetches subsequent frames e.g from a video)
    """
    def __init__(self, data_source):
        super(SequentialSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source))) #essentially: just prevents accidental shuffling

    def __len__(self):
        return len(self.data_source)

class ClusterSampler(Sampler):
    """
    Returns indices corresponding to a KMeans-clustering of the latent representations.
    (Artificial) selection bias
    """
    def __init__(self, data_source, rep_model, sample_size=10):
        super(ClusterSampler, self).__init__(data_source)
        self.data_source = data_source
        self.rep_model = rep_model
        self.rep_model.eval()
        self.reps = np.zeros((len(data_source), rep_model.latent_dim))

        with torch.no_grad():
            for i, list in tqdm(enumerate(DataLoader(self.data_source))):
                x=list[0].to("cuda").float()
                self.reps[i] = rep_model.get_encoding(x).cpu().numpy()
        np.save("reps.npy", self.reps)
        self.num_clusters = max(int(len(data_source)//(sample_size+0.1)),4)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit_predict(self.reps)
        # pca =PCA()
        # pca.fit_transform_show(X=self.reps, y=self.kmeans)


    def __iter__(self):
        return iter(np.concatenate([np.arange(len(self.data_source))[self.kmeans==i] for i in range(self.num_clusters)], axis=0))

    def __len__(self):
        return len(self.data_source)



class ClusterSamplerWithSeverity(Sampler):
    """
    Returns indices corresponding to a KMeans-clustering of the latent representations.
    (Artificial) selection bias
    """
    def __init__(self, data_source, encoding_fn, rep_model, sample_size=10, bias_severity=0.5):
        """

        :param data_source:
        :param rep_model:
        :param sample_size:
        :param bias_severity: the percentage of subsequent data that is biased. Essentially: % of data that is not sorted
        """
        super(ClusterSamplerWithSeverity, self).__init__(data_source)
        self.data_source = data_source
        self.rep_model = rep_model
        self.reps = np.zeros((len(data_source), rep_model.latent_dim))
        with torch.no_grad():
            for i, data in tqdm(enumerate(DataLoader(self.data_source))):
                x=data[0].to("cuda")
                self.reps[i] = rep_model.get_encoding(x).cpu().numpy()
        self.num_clusters = np.clip(int(len(data_source)//(sample_size+0.1)),4, 20)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit_predict(self.reps)
        self.sample_size = sample_size
        self.bias_severity = 1-bias_severity # 1-bias severity since we are selecting self.bias severity * len samples to randomize

    def __str__(self):
        return f"ClusterSamplerWithSeverity_{self.bias_severity}"
    def __iter__(self):
        full_bias = np.concatenate([np.arange(len(self.data_source))[self.kmeans==i] for i in range(self.num_clusters)], axis=0)
        shuffle_indeces = np.random.choice(np.arange(len(full_bias)), size=int(len(full_bias)*self.bias_severity), replace=False)
        to_shuffle = full_bias[shuffle_indeces]
        np.random.shuffle(to_shuffle)
        full_bias[shuffle_indeces] = to_shuffle
        return iter(full_bias)

    def __len__(self):
        return len(self.data_source)
