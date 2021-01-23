import torch
from torchvision import datasets, transforms, utils
import numpy

def get_dataset(name, train_transforms = None, test_transforms = None):
    if name == 'MNIST':
        train = datasets.MNIST('./data', train=True, download=True,transform=train_transforms)
        test = datasets.MNIST('./data', train=True, download=True,transform=test_transforms)
    elif name == 'CIFAR10':
        train = datasets.CIFAR10(root='./data', train=True,download=True,transform=train_transforms)
        test  = datasets.CIFAR10(root='./data', train=False,download=True,transform=test_transforms)
    
    return train,test
   

def get_transforms(mean, std):
    train = transforms.Compose([
                                        # transforms.Resize((28, 28)),
                                        # transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        #transforms.RandomRotation((-5.0, 5.0)),
                                        # transforms.RandomAffine((-5.0,5.0),fillcolor=1),
                                        #transforms.RandomPerspective(),
                                        transforms.ToTensor(),
                                        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                        transforms.Normalize(
                                            (mean,), (std,))
                                        ])
    test = transforms.Compose([
        #  transforms.Resize((28, 28)),
        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.ToTensor(),
        # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
        transforms.Normalize((mean,), (std,))
    ])
    return train,test


def get_dataloader(train,test,**dataloader_args):
    train = torch.utils.data.DataLoader(train, ** dataloader_args)
    test = torch.utils.data.DataLoader(test, **dataloader_args)
    return train, test


def get_data_stats(data):
    return {'Numpy Shape': data.cpu().numpy().shape,
    'Tensor Shape': data.size(),
    'min': torch.min(data),
    'max': torch.max(data),
    'mean': torch.Tensor.float(data).mean(),
    'std': torch.Tensor.float(data).std(),
    'var': torch.Tensor.float(data).var()}


#def get_data(data_loader):
#    dataiter = iter(data_loader)
#    images, labels = dataiter.next()
#    return images, lables
