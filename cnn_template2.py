import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

image_size = 28 

class tripleConv(nn.Module):
    def __init__(self):
        super(tripleConv, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)        
        return x


if __name__ == "__main__":
    train_dataset_path = "C:\\Users\\medium boss\\Downloads\\MNIST_JPG_training"
    test_dataset_path = "C:\\Users\\medium boss\\Downloads\\MNIST_JPG_testing"
    # Define image size based on your dataset
     # Change this to the size of your images

    # Define data transformations including Grayscale conversion
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with one channel
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Load custom datasets using ImageFolder for training and testing
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = tripleConv()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    train_losses = []
    val_accuracies = []
    print(model)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}\tBatch {batch_idx}/{len(train_loader)}\tLoss: {loss.item():.6f}')

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}\tTrain Loss: {avg_epoch_loss:.6f}\tValidation Accuracy: {val_accuracy:.2f}%')


    save_path = 'C:\\Users\\medium boss\\OneDrive - Cal Poly Pomona\\fall 23\\ME 4990\\kodiak.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()
    plt.show()

    print("finished training, attempting to give more info")
    # Additional information about performance
    model.eval()
    test_correct = 0
    test_total = 0

    print("in evaluation mode")

    with torch.no_grad():
        for data, target in test_loader:
            print("entered evaluation loop")
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Testing Batch {batch_idx}/{len(test_loader)}')

    test_accuracy = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
