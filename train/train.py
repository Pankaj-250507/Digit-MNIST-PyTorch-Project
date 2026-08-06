import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torchinfo import summary

from model.model import Model
from model.config import config_train
from model.config import config_model
from data.dataloaders import get_dataloaders


def train_model():

    """
    Define the training pipeline here
    """






if __name__ == "__main__" :
    model = Model(config=config_model())
    train_model()