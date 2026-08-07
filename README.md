
# Digit-MNIST-PyTorch-Project

This is a very basic deep learning project to learn CNN and ANN using PyTorch, as well as to learn git and understand the professional workflow while working on any project.

## STRUCTURE
```text
│
├── data/
│   └── dataloaders.py      # Dataset loading and preprocessing
│
├── model/
│   ├── config.py           # Model configuration
│   └── model.py            # Neural network architecture
│
├── train/
│   └── train.py            # Training loop
│
├── eval/
│   └── eval.py             # Model evaluation
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset 
- This project uses Digit-MNIST Dataset.
- It have a total of 70,000 images.
- 60,000 images for training dataset.
- 10,000 images for testing the dataset.


## WORKFLOW
1)  Fork the repository.

2) Clone your fork to your local machine 
   git clone https://github.com/<your-username>/DIGIT-MNIST-PYTORCH-PROJECT.git
   cd Digit-MNIST-PyTorch-Project

3) Create a new branch
   git checkout -b feature/<branch-name>

## Installing the required libraries
You can install the required libraries by the following command
```bash
 pip install -r requirements.txt
```
4) Complete your tasks
   You can take help of AI and the provided learning material, but if you just copy paste the code that will not be accepted.

5) After run these commands in your VS Code terminal 
   git add .
   git commit -m "<The-task-you-did>"

6) Push Your branch
   git push origin feature/<branch-name>

8) Pull the request
   - For this open your github account.
   - You will see your pushed branch.
   - Click on "COMPARE AND PULL REQUEST".
   - Submit the pull request.


## THE WORKFLOW SUMMARY

 Original Repository
        │
      Fork
        │
        ▼
   Your GitHub Fork
        │
      Clone
        │
        ▼
 Local Repository
        │
 Create Feature Branch
        │
   Make Changes
   (Complete your tasks, write the code and test it)
        │
 git add .
        │
 git commit
        │
 git push
        │
        ▼
 Pull Request
        │
        ▼
 Original Repository
