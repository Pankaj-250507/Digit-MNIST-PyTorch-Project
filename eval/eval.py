import torch 
import torch.nn as nn
from torchinfo import summary

from model.model import Model
from model.config import config_model,config_train
from data.dataloaders import get_dataloaders

def test_model():

    model_cfg=config_model()
    train_cfg=config_model()

    loaders=get_dataloaders(batch_size=train_cfg.batch_size)
    test_loader=loaders.test_loader()

    model=Model(config=model_cfg)
    model.load_state_dict(torch.load("mnist_weights.pth"))

    model.eval()

    correct=0
    total=0
    criterion=nn.CrossEntropyLoss()
    total_test_loss=0

    with torch.no_grad():
        for images,labels in test_loader:
            outputs=model(images)
            loss=criterion(outputs,labels)
            total_test_loss+=loss.item()

            _, predicted=torch.max(outputs.data,1)

            total+=labels.size(0)

            correct+=(predicted==labels).sum().item()

    avg_test_loss=total_test_loss/len(test_loader)

    accuracy=100*correct/total

    print(f"\n--Evaluation Results ---")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy :{accuracy:.2f}% ({correct}/{total} correct images)")

if __name__=="__main__":
    test_model()