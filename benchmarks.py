from domain_datasets import *
import torch.utils.data as data
from abc import ABC, abstractmethod

class Bench(ABC):
    def __init__(self, bias=True):
        self.bias = bias

    @abstractmethod
    def get_ind(self):
        pass

    @abstractmethod
    def get_ood(self):
        pass

    @abstractmethod
    def transformed_ind(self, transform):
        pass
    @abstractmethod
    def eval_sample(self, x, y):
        pass

class PolypBench(Bench):
    def __init__(self):
        super(PolypBench, self).__init__()
        pass

    def get_ind(self):
        pass

    def get_ood(self):
        pass

    def transformed_ind(self, transform):
        pass

    def eval_sample(self, x, y):
        pass

class NICOBench(Bench):
    def __init__(self):
        super(NICOBench, self).__init__()
        pass

    def get_ind(self):
        pass

    def get_ood(self):
        pass

    def transformed_ind(self, transform):
        pass

    def eval_sample(self, x, y):
        pass

class CamelyonBench(Bench):
    def __init__(self):
        super(CamelyonBench, self).__init__()
        pass

    def get_ind(self):
        pass

    def get_ood(self):
        pass

    def transformed_ind(self, transform):
        pass

    def eval_sample(self, x, y):
        pass

class CIFAR10Bench(Bench):
    def __init__(self):
        super(CIFAR10Bench, self).__init__()
        pass

    def get_ind(self):
        pass

    def get_ood(self):
        pass

    def transformed_ind(self, transform):
        pass

    def eval_sample(self, x, y):
        pass

class NjordBench(Bench):
    def __init__(self):
        super(NjordBench, self).__init__()
        pass

    def get_ind(self):
        pass

    def get_ood(self):
        pass

    def transformed_ind(self, transform):
        pass

    def eval_sample(self, x, y):
        pass