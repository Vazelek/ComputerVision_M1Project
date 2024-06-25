import os
import shutil
import sys
import time

import numpy as np
import pickle
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import torch
import keras
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import data_utils
import custom_model
import models
import train_evaluate
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_PATH)


def display_help():
    print(">>> python main.py [OPTIONS]\n")
    print("\t-r, --reload\tRecreate split_data directory and data cache and retrain models")
    print("\t-s,  --small\t\tUse of a smaller dataset (for testing purposes)")
    print("\t-h,  --help\t\t\tDisplay this help")


if __name__ == "__main__":
    data_dir = '../data'
    custom_images_dir = '../CustomImages'
    data_dirs = [data_dir, custom_images_dir]
    large_split_data_dir = '../split_data'
    small_split_data_dir = '../small_split_data'
    models_dir = '../models'
    selected_data_dir = large_split_data_dir

    reload = False
    small_data = False
    if len(sys.argv) >= 4:
        print(f"Too many arguments ({len(sys.argv) - 1}/2)\n")
        display_help()
    elif len(sys.argv) == 2:
        if sys.argv[1] == "-r" or sys.argv[1] == "--reload":
            reload = True
        elif sys.argv[1] == "-s" or sys.argv[1] == "--small":
            small_data = True
            selected_data_dir = small_split_data_dir
        else:
            display_help()
    elif len(sys.argv) == 3:
        if ((sys.argv[1] == "-r" or sys.argv[1] == "--reload") and (sys.argv[2] == "-s" or sys.argv[2] == "--small")) \
                or ((sys.argv[2] == "-r" or sys.argv[2] == "--reload") and (sys.argv[1] == "-s" or sys.argv[1] == "--small")):
            reload = True
            small_data = True
            selected_data_dir = small_split_data_dir
        else:
            display_help()

    train_dir = selected_data_dir + '/train'
    val_dir = selected_data_dir + '/validation'

    if small_data and (reload or not os.path.isdir(small_split_data_dir)):
        print("Splitting small data")
        if os.path.isdir(small_split_data_dir):
            shutil.rmtree(small_split_data_dir)
        data_utils.split_dataset(data_dirs, train_dir, val_dir, val_ratio=0.2, small_data=True)
    elif not small_data and (reload or not os.path.isdir(large_split_data_dir)):
        print("Splitting data")
        if os.path.isdir(large_split_data_dir):
            shutil.rmtree(large_split_data_dir)
        data_utils.split_dataset(data_dirs, train_dir, val_dir, val_ratio=0.2)

    dataset_cache = 'dataset_cache' + ('_small' if small_data else '') + '.pkl'
    if os.path.exists(dataset_cache) and not reload:
        with open(dataset_cache, 'rb') as f:
            train_images, train_labels, test_images, test_labels, class_names = pickle.load(f)
        print("Loaded dataset from cache.")
    else:
        print("Process dataset")
        train_images, train_labels, class_names = data_utils.load_dataset(train_dir)
        test_images, test_labels, _ = data_utils.load_dataset(val_dir)
        with open(dataset_cache, 'wb') as f:
            pickle.dump((train_images, train_labels, test_images, test_labels, class_names), f)
        print("Saved dataset to cache.")

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    input_shape = (64, 64, 3)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_to_train = {
        "custom_model": custom_model.create_custom_model(input_shape, num_classes),
        "resnet18": models.get_model("resnet18", num_classes),
        # "alexnet": models.get_model("alexnet", num_classes), # horizontal results for all scores (only one class is detected)
        # "vgg16": models.get_model("vgg16", num_classes), # takes too long to train and has bad results
        "densenet121": models.get_model("densenet121", num_classes),
        "mobilenet_v2": models.get_model("mobilenet_v2", num_classes)
    }

    all_train_losses = {model_name: [] for model_name in models_to_train}
    all_train_accuracies = {model_name: [] for model_name in models_to_train}
    all_val_losses = {model_name: [] for model_name in models_to_train}
    all_val_accuracies = {model_name: [] for model_name in models_to_train}
    all_val_f1_scores = {model_name: [] for model_name in models_to_train}
    all_train_time = {model_name: [] for model_name in models_to_train}
    all_confusion_matrix = {model_name: [] for model_name in models_to_train}

    num_epochs = 10
    criterion = torch.nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = TensorDataset(torch.tensor(train_images).permute(0, 3, 1, 2), torch.tensor(train_labels))
    test_dataset = TensorDataset(torch.tensor(test_images).permute(0, 3, 1, 2), torch.tensor(test_labels))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    models_dir = '../models'
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    for model_name, model in models_to_train.items():
        print(f"Training {model_name}...")
        if model_name == "custom_model":
            save_file = os.path.join(models_dir, model_name + ".keras")
        else:
            save_file = os.path.join(models_dir, model_name + ".pth")

        if os.path.isfile(save_file) and not reload:
            if model_name == "custom_model":
                model = keras.models.load_model(save_file)
                print(f"Model {model_name} loaded from models/")
            else:
                model.load_state_dict(torch.load(save_file))
                print(f"Model {model_name} loaded from models/")
        else:
            if os.path.isfile(save_file):
                os.remove(save_file)

            start_time = time.time()

            if model_name == "custom_model":
                for epoch in range(num_epochs):
                    history = model.fit(train_images, train_labels, epochs=1,
                                        validation_data=(test_images, test_labels), verbose=0)

                    train_loss = history.history['loss'][-1]
                    train_accuracy = history.history['accuracy'][-1]
                    val_loss = history.history['val_loss'][-1]
                    val_accuracy = history.history['val_accuracy'][-1]

                    val_predictions = np.argmax(model.predict(test_images), axis=-1)

                    val_f1_score = f1_score(test_labels, val_predictions, average='weighted', zero_division=1)

                    if epoch + 1 == num_epochs:
                        cm = confusion_matrix(test_labels, val_predictions)
                        all_confusion_matrix[model_name] = cm

                    all_train_losses[model_name].append(train_loss)
                    all_train_accuracies[model_name].append(train_accuracy)
                    all_val_losses[model_name].append(val_loss)
                    all_val_accuracies[model_name].append(val_accuracy)
                    all_val_f1_scores[model_name].append(val_f1_score)

                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1_score:.4f}")

                model.save(save_file)
                print(f"Model {model_name} saved in models/")
            else:
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                for epoch in range(num_epochs):
                    train_loss, train_accuracy = train_evaluate.train(model, train_loader, criterion, optimizer, device)

                    val_loss, val_accuracy, val_f1_score, cm = train_evaluate.validate(model, test_loader, criterion, device, class_names)

                    all_train_losses[model_name].append(train_loss)
                    all_train_accuracies[model_name].append(train_accuracy)
                    all_val_losses[model_name].append(val_loss)
                    all_val_accuracies[model_name].append(val_accuracy)
                    all_val_f1_scores[model_name].append(val_f1_score)

                    if epoch + 1 == num_epochs:
                        all_confusion_matrix[model_name] = cm

                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1_score:.4f}")

                torch.save(model.state_dict(), save_file)
                print(f"Model {model_name} saved in models/")

            end_time = time.time()

            train_time = end_time - start_time
            all_train_time[model_name] = train_time

            print(f"{model_name} training finished in {train_time} seconds")

    save_dir = '../plots'
    train_evaluate.plot_metrics(num_epochs, all_train_losses, all_train_accuracies, all_val_losses, all_val_accuracies, all_val_f1_scores, all_train_time, all_confusion_matrix, models_to_train.keys(), save_dir, class_names)
