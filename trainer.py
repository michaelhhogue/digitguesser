import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from neural_network import NeuralNetwork

# Hyperparameters
learning_rate = 0.01
batch_size = 64
epochs = 20

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"({((batch + 1) * len(X)):>6d}/{size:<6d}) | Loss: {loss.item()}")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    # Define model
    model = NeuralNetwork(True)

    load_existing_model = input("Would you like to load an existing model? (Y/n): ")
    if load_existing_model.upper() == 'Y':
        model_file_name = input("Enter the model file name: ")
        model.load_state_dict(torch.load(model_file_name))

    print("Loading transform composition...")

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

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Perform training and testing loop epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
    print("Done!")

    save_file_name = input("Save model as (filename): ")
    torch.save(model.state_dict(), save_file_name)
