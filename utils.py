import torch
from torch.utils import data
import torchvision.transforms as transforms
import torch.nn as nn

def repeat_dataset_with_transforms(dataset, transforms_list):
    """this method returns a new dataset object that inherits all properties from the dataset parameter,
    but each new dataset is transformed by one of the transforms in transforms_list
    """
    class NewDataset(data.Dataset):
        def __init__(self, dataset, transforms):
            self.dataset = dataset
            self.transforms = transforms

        def __getitem__(self, index):
            x, y = self.dataset[index]
            image, mask = self.train_transforms(image=x, mask=y).values()
            return image, mask

        def __len__(self):
            return len(self.dataset)

    return data.ConcatDataset([NewDataset(dataset, transform) for transform in transforms_list])



class WrappedResnet(nn.Module):
    def __init__(self, model, input_size=32):
        super().__init__()
        self.model = model
        self.latent_dim = self.get_encoding_size(input_size)

    def get_encoding_size(self, input_size):
        dummy = torch.zeros((1, 3, input_size, input_size)).to("cuda")
        return self.get_encoding(dummy).shape[-1]

    def get_encoding(self, X):
        return torch.nn.Sequential(*list(self.model.children())[:-2])(X).flatten(1)

    def forward(self, x):
        return self.model(x)



#write a method that takes an object as argument and returns a class that extends that object


class ArgumentIterator:
    #add arguments to an iterator for use in parallell processing
    def __init__(self, iterable, variables):
        self.iterable = iterable
        self.index = 0
        self.variables = variables

    def __next__(self):
        if self.index >= len(self.iterable):
            raise StopIteration
        else:
            self.index += 1
            return *self.iterable[self.index-1], *self.variables

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)
