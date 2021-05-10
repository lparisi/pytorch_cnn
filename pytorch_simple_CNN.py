#imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')

#set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
IN_CHANNEL = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 5


class CNN(nn.Module):
    def __init__(self, in_channels =1, num_classes = NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(IN_CHANNEL, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

#Load Data
train_dataset =  datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset =  datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

#Initialize network
model = CNN().to(device=device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#Train Network
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        #Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        #forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or Adam step
        optimizer.step()

#Check accuracy on training & test to see how good our model is

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on the training data...")
    else:
        print("Checking accuracy on the test data...")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")