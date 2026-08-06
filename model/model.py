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
        ...

    def forward():
        ...
    



if __name__=='__main__':
    model = Model(config = config_model())