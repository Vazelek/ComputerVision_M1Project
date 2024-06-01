import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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


def plot_metrics(num_epochs, all_train_losses, all_val_losses, all_val_accuracies, all_val_f1_scores, model_names,
                 save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(20, 12))

    plt.subplot(3, 1, 1)
    for model_name in model_names:
        plt.plot(epochs, all_train_losses[model_name], label=f'{model_name} Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))

    plt.subplot(3, 1, 2)
    for model_name in model_names:
        plt.plot(epochs, all_val_losses[model_name], label=f'{model_name} Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'validation_loss.png'))

    plt.subplot(3, 1, 3)
    for model_name in model_names:
        plt.plot(epochs, all_val_accuracies[model_name], label=f'{model_name} Validation Accuracy')
        plt.plot(epochs, all_val_f1_scores[model_name], '--', label=f'{model_name} Validation F1 Score')
    plt.title('Validation Accuracy and F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'validation_accuracy_f1.png'))

    plt.show()
