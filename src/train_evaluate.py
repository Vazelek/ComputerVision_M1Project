import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).long()  # Convert labels to LongTensor

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    return train_loss, train_accuracy


def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()  # Convert labels to LongTensor

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_loss = running_loss / len(test_loader)
    val_accuracy = correct / total

    val_f1_score = f1_score(all_labels, all_preds, average='weighted')
    return val_loss, val_accuracy, val_f1_score


def plot_metrics(num_epochs, train_losses, train_accuracies, val_losses, val_accuracies, val_f1_scores, train_times, model_names, save_dir):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(5, 5))
    for model_name in model_names:
        if len(train_losses[model_name]) == 0:
            continue

        plt.plot(epochs, train_losses[model_name], label=f'{model_name} Train Loss')

    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    plt.show()

    plt.figure(figsize=(10, 5))
    count = 0
    for model_name in model_names:
        if len(train_losses[model_name]) == 0:
            continue

        count += 1

        plt.plot(epochs, train_accuracies[model_name], label=f'{model_name} Train Accuracy')

    if count == 0:
        return

    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_accuracy.png'))
    plt.show()

    plt.figure(figsize=(10, 5))
    for model_name in model_names:
        if len(train_losses[model_name]) == 0:
            continue

        plt.plot(epochs, val_losses[model_name], label=f'{model_name} Validation Loss')

    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'validation_loss.png'))
    plt.show()

    plt.figure(figsize=(10, 5))
    for model_name in model_names:
        if len(train_losses[model_name]) == 0:
            continue

        plt.plot(epochs, val_accuracies[model_name], label=f'{model_name} Validation Accuracy')

    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'validation_accuracy.png'))
    plt.show()

    plt.figure(figsize=(10, 5))
    for model_name in model_names:
        if len(train_losses[model_name]) == 0:
            continue

        plt.plot(epochs, val_f1_scores[model_name], '--', label=f'{model_name} Validation F1 Score')

    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'validation_f1score.png'))
    plt.show()

    plt.figure(figsize=(10, 5))
    for model_name in model_names:
        if len(train_losses[model_name]) == 0:
            continue

        plt.bar(model_name, train_times[model_name], color='maroon', width=0.4)

    plt.title('Training time')
    plt.xlabel('Model')
    plt.ylabel('Train time')
    plt.savefig(os.path.join(save_dir, 'train_time.png'))
    plt.show()
