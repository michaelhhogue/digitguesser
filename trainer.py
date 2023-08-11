import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

# Hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 5

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits

if __name__ == "__main__":
    print("Loading transform composition...")

    # Convert to tensor and normalize with mean=0.5, std=0.5
    # for the grayscale pixel intensities
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

    print("Loading MNIST dataset...")

    # Load the MNIST training data
    train_data = datasets.MNIST(
            root='MNIST_data/',
            train=True,
            download=True,
            transform=transform)

    # Load the MNIST test data
    test_data = datasets.MNIST(
            root='MNIST_data/',
            train=False,
            download=True,
            transform=transform)

    # Create the data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Define model
    model = NeuralNetwork()
