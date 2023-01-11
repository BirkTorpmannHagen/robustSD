import torch
from torch.utils.data import Sampler, DataLoader
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

class ClassOrderSampler(Sampler):
    """
    Sampler that splits the data_source into classes, returns indexes in order of class
    Induces selection bias via label shift
    """
    def __init__(self, data_source):
        super(ClassOrderSampler, self).__init__(data_source)
        self.data_source = data_source
        self.indices = [[] for i in data_source.classes]

        #initial pass to sort the indices by class
        for i, (x,y,_) in enumerate(data_source):
            self.indices[y].append(i)


    def __iter__(self):
        return sum(self.indices, [])


class SequentialSampler(Sampler):
    """
    Samples sequentially from the dataset (assuming the dataloader fetches subsequent frames e.g from a video)
    """
    def __init__(self, data_source):
        super(SequentialSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source))) #essentially: just prevents accidental shuffling


class ClusterSampler(Sampler):
    """
    Returns indices corresponding to a KMeans-clustering of the latent representations.
    (Artificial) selection bias
    """
    def __init__(self, data_source, rep_model):
        super(ClusterSampler, self).__init__(data_source)
        self.data_source = data_source
        self.rep_model = rep_model
        self.reps = np.zeros((len(data_source), rep_model.latent_dim))
        with torch.no_grad():
            for i, (x,y,_) in tqdm(enumerate(DataLoader(self.data_source))):
                x=x.to("cuda")
                self.reps[i] = rep_model.encode(x)[0].cpu().numpy()
        self.kmeans = KMeans(n_clusters=10, random_state=0).fit_predict(self.reps)


    def __iter__(self):
        return iter(np.concatenate([np.arange(len(self.data_source))[self.kmeans==i] for i in range(10)], axis=0))