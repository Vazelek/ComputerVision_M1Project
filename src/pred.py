import os
import numpy as np
import tensorflow as tf
import keras
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import models

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_PATH)


def preprocess_image(file_path, img_size=(64, 64)):
    try:
        img = Image.open(file_path).convert('RGB')
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2
        img = img.crop((left, top, right, bottom))
        img = img.resize(img_size)
        img = np.array(img, dtype=np.float32) / 255.0
        return img
    except OSError as e:
        print(f"Skipping corrupted image: {file_path} - {e}")
        return None


def load_image_dataset(image_dir, img_size, all_class_names):
    file_paths = []
    labels = []
    present_class_indices = []

    for class_index, class_name in enumerate(all_class_names):
        class_dir = os.path.join(image_dir, class_name)
        if os.path.isdir(class_dir):
            class_found = False
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if os.path.isfile(file_path) and file_path.endswith(('.png', '.jpg', '.jpeg')):
                    img = preprocess_image(file_path, img_size)
                    if img is not None:
                        file_paths.append(img)
                        labels.append(class_index)
                        class_found = True
            if class_found:
                present_class_indices.append(class_index)

    images = np.array(file_paths)
    labels = np.array(labels)

    return images, labels, present_class_indices


def predict_with_keras_model(model, images):
    return model.predict(images)


def predict_with_pytorch_model(model, images):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_images = torch.stack([transform(image) for image in images])
    with torch.no_grad():
        outputs = model(tensor_images)
    return outputs.numpy()


def predict_images(image_dir, full_data_dir):
    img_size = (64, 64)

    # Get the full set of class names from the complete dataset directory
    all_class_names = os.listdir(full_data_dir)
    all_class_names.sort()

    # Load images and labels from the custom image directory
    images, labels, present_class_indices = load_image_dataset(image_dir, img_size, all_class_names)

    models_dir = '../models'

    for filename in os.listdir(models_dir):
        file_split = filename.split(".")
        model_type = file_split[0]
        model = None

        if model_type == "custom_model":
            model = keras.models.load_model(os.path.join(models_dir, filename))
            print(f"Model {model_type} loaded from models/")
            predictions = predict_with_keras_model(model, images)
        else:
            model = models.get_model(model_type, len(all_class_names))
            model.load_state_dict(torch.load(os.path.join(models_dir, filename)))
            print(f"Model {model_type} loaded from models/")
            predictions = predict_with_pytorch_model(model, images)

        fig = plt.figure(figsize=(16, 16))
        fig.suptitle(f"Predictions for the {model_type} model")

        for i, class_index in enumerate(present_class_indices):
            ax = fig.add_subplot(4, 4, i + 1)
            class_indices = np.where(labels == class_index)[0]
            if len(class_indices) == 0:
                continue
            img_idx = class_indices[0]
            ax.imshow(images[img_idx])
            predicted_label = np.argmax(predictions[img_idx])
            true_label = labels[img_idx]
            ax.set_title("Predicted: {}\nTrue:{}".format(all_class_names[predicted_label], all_class_names[true_label]))
            ax.axis('off')

        plt.show()


if __name__ == "__main__":
    predict_images("../CustomImages", "../data")
