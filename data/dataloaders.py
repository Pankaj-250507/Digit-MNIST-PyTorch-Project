'''
This module contains the dataloaders for the MNIST dataset based on the training config.
'''

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from model.config import config_train

train_transform = transforms.Compose([
    transforms.RandAugment(num_ops = 3, magnitude = 8,interpolation = transforms.InterpolationMode.BILINEAR), 
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]
    )

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

test_dataset = MNIST(
        root="./data",
        train = False,
        download=True,
        transform=test_transform
)

class get_dataloaders:
    '''
    This class is used to get the dataloaders for training and testing datasets.
    '''
    def __init__(self, config : config_train, train_dataset = train_dataset, test_dataset = test_dataset):
        '''
        Initializes the get_dataloaders class with the given configuration and datasets.
        Args : 
            config (config_train): The configuration for the dataloaders.
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The testing dataset.
        '''
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train_loader(self):
        '''
        Returns the training dataloader.
        Returns:
            DataLoader: The training dataloader.
        '''
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory
        )
    def test_loader(self):
        '''
        Returns the testing dataloader.
        Returns:
            DataLoader: The testing dataloader.
        '''
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory
        )

if __name__ == "__main__":
    # basic tests
    config = config_train(batch_size=64)
    dataloaders = get_dataloaders(config)
    train_loader = dataloaders.train_loader()
    test_loader = dataloaders.test_loader()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")