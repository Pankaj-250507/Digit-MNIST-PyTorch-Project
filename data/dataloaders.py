from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# This file handles loading and preprocessing the MNIST dataset.
# It prepares the images and labels using PyTorch DataLoader.
# You can freely experiment with transformations and batch sizes here.

train_transform = transforms.Compose(
    ''' Apply the transformations on dataset here'''

    )

test_transform = transforms.Compose(

)

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

    def __init__(self):
        '''
        Initialize the required arguments here
        '''

    def train_loader(self):
        '''
        Implement the training dataset loader here'''

    def test_loader(self):

        ...

    