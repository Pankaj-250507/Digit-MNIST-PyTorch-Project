from dataclasses import dataclass

@dataclass
class config_model:
    num_classes :int = 10
    in_channels=1

    conv1_filters:int = 32
    conv2_filters=64
    kernel_size=3
    hidden_dim=128

    # Define the configurations required in model architecture

@dataclass
class config_train:
    #dataloaders parameters
  
    batch_size:int = 64
    
    # training paramerters 
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs=5
