import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_size = 28*28, hidden_size1 = 128, hidden_size2 = 64, output_size = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) # First hidden layer / Input layer
        self.relu1 = nn.ReLU() # activation func. for first layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) # Second hidden layer
        self.relu2 = nn.ReLU() # activation func. for second layer
        self.fc3 = nn.Linear(hidden_size2, output_size) # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten 28x28 image into 784-dim vector
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epoch = 20

for epoch in range(num_epoch):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)


        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}")

correct = 0
total = 0

with torch.no_grad(): # disable gradient computation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Get predicted class (highest probability)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
print(f'Accuracy: {100 * correct / total:.2f}%')
