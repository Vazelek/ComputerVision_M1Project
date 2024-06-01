import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from CustomDataset import CustomDataset
import pickle
import matplotlib.pyplot as plt


def convert_to_rgb(image):
    return image.convert('RGB')


dataset_cache = 'dataset_cache_small.pkl'
with open(dataset_cache, 'rb') as f:
    train_images, train_labels, test_images, test_labels, class_names = pickle.load(f)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Transformations for the training and validation data
transform = transforms.Compose([
    transforms.ToPILImage(),  # Ensure image is converted to PIL Image
    transforms.Resize((224, 224)),
    transforms.Lambda(convert_to_rgb),  # Convert image to RGB using a defined function
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    # Assuming train_images, train_labels, test_images, test_labels are numpy arrays or similar
    global train_images, train_labels, test_images, test_labels, class_names

    # Create custom datasets
    train_dataset = CustomDataset(train_images, train_labels, transform=transform)
    test_dataset = CustomDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Define a list of models to compare
    model_names = ["resnet18", "alexnet", "vgg16", "densenet121", "mobilenet_v2"]
    model_constructors = {
        "resnet18": models.resnet18,
        "alexnet": models.alexnet,
        "vgg16": models.vgg16,
        "densenet121": models.densenet121,
        "mobilenet_v2": models.mobilenet_v2
    }
    weights = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "alexnet": models.AlexNet_Weights.DEFAULT,
        "vgg16": models.VGG16_Weights.DEFAULT,
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "mobilenet_v2": models.MobileNet_V2_Weights.DEFAULT
    }

    # Dictionaries to store metrics for each model
    all_train_losses = {name: [] for name in model_names}
    all_val_losses = {name: [] for name in model_names}
    all_val_accuracies = {name: [] for name in model_names}
    all_val_f1_scores = {name: [] for name in model_names}

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training function
    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Convert labels to LongTensor
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        return running_loss / len(train_loader)

    # Validation function
    def validate(model, test_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Convert labels to LongTensor
                labels = labels.long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return running_loss / len(test_loader), accuracy, f1

    # Training and validation for each model
    num_epochs = 10
    for model_name in model_names:
        print(f"Training {model_name}...")
        model = model_constructors[model_name](weights=weights[model_name])

        # Modify the final layer to match the number of classes in your dataset
        if model_name.startswith("resnet") or model_name.startswith("densenet"):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(class_names))
        elif model_name.startswith("alexnet") or model_name.startswith("vgg"):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, len(class_names))
        elif model_name.startswith("mobilenet"):
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, len(class_names))

        model = model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(num_epochs):
            print("-----------")
            print(f'Epoch {epoch + 1}/{num_epochs}')
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy, val_f1 = validate(model, test_loader, criterion, device)

            all_train_losses[model_name].append(train_loss)
            all_val_losses[model_name].append(val_loss)
            all_val_accuracies[model_name].append(val_accuracy)
            all_val_f1_scores[model_name].append(val_f1)

            print(f'Train Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}')

        print(f'Training complete for {model_name}')

    # Plot the metrics
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(20, 12))

    plt.subplot(3, 1, 1)
    for model_name in model_names:
        plt.plot(epochs, all_train_losses[model_name], label=f'{model_name} Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    for model_name in model_names:
        plt.plot(epochs, all_val_losses[model_name], label=f'{model_name} Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 3)
    for model_name in model_names:
        plt.plot(epochs, all_val_accuracies[model_name], label=f'{model_name} Validation Accuracy')
        plt.plot(epochs, all_val_f1_scores[model_name], '--', label=f'{model_name} Validation F1 Score')
    plt.title('Validation Accuracy and F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
