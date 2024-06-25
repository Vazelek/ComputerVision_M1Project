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
import sys

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


def load_image_dataset(image_dir, img_size):
    images = []
    file_paths = []

    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        if os.path.isfile(file_path) and file_path.endswith(('.png', '.jpg', '.jpeg')):
            img = preprocess_image(file_path, img_size)
            if img is not None:
                images.append(img)
                file_paths.append(file_path)

    images = np.array(images)
    return images, file_paths


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

    # Load images from the custom image directory
    images, file_paths = load_image_dataset(image_dir, img_size)

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

        for i, img_path in enumerate(file_paths):
            ax = fig.add_subplot(8, 8, i + 1)
            ax.imshow(images[i])
            predicted_label = np.argmax(predictions[i])
            ax.set_title(all_class_names[predicted_label])
            ax.axis('off')

            if i >= 63:
                break

        plt.show()


def display_help():
    print(">>> python pred.py <directory_name>\n")
    print("\t<directory_name>\tThe directory where are the images you want to predict there class")
    exit(1)


if __name__ == "__main__":
    test_dir = ""
    if len(sys.argv) > 2:
        print(f"Too many arguments ({len(sys.argv) - 1}/1)\n")
        display_help()
    elif len(sys.argv) == 2:
        if os.path.exists("../" + sys.argv[1]):
            test_dir = "../" + sys.argv[1]
        else:
            print("\"" + sys.argv[1] + "\": Directory does not exist\n")
            display_help()
    else:
        print("No directory name provided\n")
        display_help()

    print("Testing with directory " + test_dir.split("/")[-1])
    predict_images(test_dir, "../data")
