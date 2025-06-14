
import torch.nn as nn
import torch.nn.functional as F


def load_cifar10_data(root_dir, 
                      batch_size=None, 
                      num_workers=None):
    """
    Load CIFAR-10 dataset from the specified root directory.
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    # Data augmentation for training data
    transform =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))])
    
    full_dataset = datasets.CIFAR10(root=root_dir, train=True, 
                                    download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root_dir, train=False,
                                     download=True, transform=transform)
    
    # Split the training dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def data_augmentation(data_dir, batch_size=None, num_workers=None):
    from torch.utils.data import DataLoader
    from torchvision import datasets

    # Data augmentation for training data
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),    # Randomly flip image horizontally 50% of the time
        transforms.RandomRotation(degrees=15),     # Randomly rotate image within ±15 degrees
        transforms.RandomCrop(32, padding=4),      # Random crop with 4-pixel padding around edges
        transforms.ToTensor(),                      # Convert to tensor (range 0-1)
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # Normalize with CIFAR-10 mean and std
                            (0.2023, 0.1994, 0.2010))
    ])

    # For test/validation, no augmentation, just normalization
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                    download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=val_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
# ----------------------------
# 5. Model training comes next
# ----------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        import torch.nn as nn
        import torch.nn.functional as F

        super(SimpleCNN, self).__init__()
        # Conv Layer 1: input channels=3 (RGB), output channels=32, kernel size=3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Conv Layer 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # after 2 pooling layers, image size reduced from 32 to 8
        self.fc2 = nn.Linear(512, 10)  # 10 output classes
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))   # Conv1 + ReLU
        x = self.pool(x)            # Pooling 1
        x = F.relu(self.conv2(x))   # Conv2 + ReLU
        x = self.pool(x)            # Pooling 2
        x = x.view(-1, 64 * 8 * 8) # Flatten
        x = F.relu(self.fc1(x))     # Fully connected 1 + ReLU
        x = self.dropout(x)         # Dropout
        x = self.fc2(x)             # Fully connected 2 (output layer)
        return x


def train(model, train_loader, val_loader, criterion,
          optimizer, device, num_epochs=10, writer=None,
          excel_path="training_metrics.xlsx"):
    
    import torch
    import pandas as pd
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_preds': [],
        'val_labels': []
    }

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0

        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Val", val_acc, epoch)

        # Save to history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_preds'].append(val_preds)
        history['val_labels'].append(val_labels)

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Export metrics to Excel
    df_history = pd.DataFrame(history)
    df_history.to_excel(excel_path, index=False)

    print(f"\n✅ Training complete. Metrics saved to: {excel_path}")
    return history

def generate_metrices(history):
    """
    Generate a dictionary of metrics from the training history.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd
    import numpy as np

    preds = np.concatenate(history['val_preds'])
    labels = np.concatenate(history['val_labels'])

    # Classification report
    report = classification_report(labels, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("results/classification_report.csv", index=True)
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv("results/confusion_matrix.csv", index=True)

    return report_df, cm_df

def plot_training(history):

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_metrics.png")
    print("Training metrics plot saved as 'results/training_metrics.png'")
    plt.show()


if __name__ == "__main__":

    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets, transforms

    batch_size = 64
    num_workers = 2
    num_epochs = 20

    root_dir = './data'  # Change this to your desired data directory
    path_losses = "results/training_metrics.xlsx"
    writer = SummaryWriter(log_dir="runs/cifar10_experiment")
    train_loader, val_loader, test_loader = load_cifar10_data(root_dir,
                                                              batch_size=batch_size,
                                                              num_workers=num_workers)
    
    aug_train_loader, aug_val_loader, aug_test_loader = data_augmentation(
        root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"-----------------------------------------------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"-----------------------------------------------")
    
    # Define your model, loaders, loss, optimizer
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = train(model, 
                    aug_train_loader, 
                    aug_val_loader, 
                    criterion, optimizer, device, num_epochs, writer, excel_path=path_losses)
    writer.close()

    # %% Generate metrics
    report_df, cm_df = generate_metrices(history)

    # %% save the model

    model_path = "models/cifar10_cnn.pth"
    checkpoint = {
        'epoch': history['epoch'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'num_epochs': num_epochs,
        'model_name': 'SimpleCNN',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'loss_function': 'CrossEntropyLoss',
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, model_path)

    # %% 
    plot_training(history)

    #  when the training is done, run "tensorboard --logdir=runs"
