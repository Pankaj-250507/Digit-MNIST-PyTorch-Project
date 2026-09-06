import torch
import torch.nn as nn
import torchinfo
from model.config import config_model


class Model(nn.Module):

    '''
    Construct the model architecture. 
    Use CNN, then flatten the tensor and then apply ANN for classification
    '''

    def __init__(self, config):
        '''
        Initialization with configuration partamters
        '''
        super(Model,self).__init__()

        self.conv1=nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.conv1_filters,
            kernel_size=config.kernel_size,
            padding=1
        )

        self.conv2=nn.Conv2d(
            in_channels=config.conv1_filters,
            out_channels=config.conv2_filters,
            kernel_size=config.kernel_size,
            padding=1
        )

        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

        self.relu-nn.Relu()

        self.flatten=nn.Flatten()

        flattened_size=config.conv2_filters*7*7

        self.fc1=nn.Linear(flattened_size,config.hidden_dim)

        self.fc2=nn.Linear(config.hidden_dim,config.num_classes)

    def forward(self,x):

        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.flatten(x)

        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)

        return x
    



