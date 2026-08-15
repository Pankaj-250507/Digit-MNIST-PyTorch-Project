from dataclasses import dataclass
import os
import torch

@dataclass
class config_model:
    num_classes :int = ...
    conv1:int = ...
    ...

    # Define the configurations required in model architecture

@dataclass
class config_train:
    # Dataloaders parameters
    batch_size:int = ...
    num_workers : int = os.cpu_count()//2  # Number of parallel worker processes for data loading
    persistent_workers : bool = bool(num_workers > 0) # If true it will reuse workers for multiple epochs, more efficient
    prefetch_factor : int = 2 # how many batches each worker shld prefetch
    pin_memory : bool = bool(torch.cuda.is_available()) # pins memory to GPU if available, improves transfer speed
    
    # training paramerters 
    learning_rate: float = ...
    weight_decay: float = ...
    ...
