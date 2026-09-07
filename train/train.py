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
    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device :{device}")
    model_cfg=config_model()
    train_cfg=config_train()

    loaders=get_dataloaders(batch_size=train_cfg.batch_size)
    train_loader=loaders.train_loader()

    model=Model(config=model_cfg).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(
        model.paramters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay
    )

    model.train()
    for epoch in range(train_cfg.epochs):
        total_loss=0
        loop=tqdm(train_loader,desc=f"Epoch [{epoch+1}/{train_cfg.epochs}]")

        for images, labels in loop:
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            loss.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss=total_loss/len(train_loader)
        print(f"Epoch {epoch+1} Finished | Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "mnist_weights.pth")
    print("Training complete! Model weights saved to mnist_weights.pth")

if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model=Model(config=config_model()).to(device)

    summary(model, input_size=(config_train().batch_size, config_model().in_channels,28,28))

    train_model()


        
        






