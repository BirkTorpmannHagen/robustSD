import torch
from torch.utils import data


def wrap_model(model):
    class Wrapped:
        def __init__(self):
            self.model = model
            self.latent_dim = self.get_encoding_size()
            self.criterion = torch.nn.CrossEntropyLoss()

        def forward(self, X):
            return self.model(X)

        def __call__(self, X):
            return  self.model(X)

        def compute_loss(self, x,y):
            return self.criterion(self.forward(x),y)

        def get_encoding_size(self):
            dummy = torch.zeros((1, 3, 512, 512)).to("cuda")
            return torch.nn.Sequential(*list(self.model.children())[:-1])(dummy).flatten(1).shape[-1]

        def encode(self, X):
            return torch.nn.Sequential(*list(self.model.children())[:-1])(X).flatten(1)
    return Wrapped()


#write a method that takes an object as argument and returns a class that extends that object
def wrap_dataset(dataset):
    """
    dumb utility function to make testing easier. standardizes datasets so that it works easier with the models and trainers
    :param dataset:
    :return:
    """
    class NewDataset(data.Dataset):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

        def __getitem__(self, index):
            image, label = self.dataset[index]
            if image.shape[0]==1:
                image = image.repeat(3,1,1)
            return image, label, 0

        def __len__(self):
            return len(self.dataset)

    return NewDataset(dataset)

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
