from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# This file handles loading and preprocessing the MNIST dataset.
# It prepares the images and labels using PyTorch DataLoader.
# You can freely experiment with transformations and batch sizes here.

train_transform = transforms.Compose(
    
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ]

    )

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
]

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

    def __init__(self,batch_size=64):
        
        self.batch_size=batch_size

    def train_loader(self):
        
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )
    def test_loader(self):

        return DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True
        )

        ...

    