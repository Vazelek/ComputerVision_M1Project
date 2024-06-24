import os
import shutil
import random
import pickle
from PIL import Image
import numpy as np

SMALL_RATIO = 0.05
LIMIT = 0  # classes - for test purposes


def split_dataset(data_dirs, train_dir, val_dir, val_ratio=0.2, small_data=False):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    is_custom_images = False
    for data_dir in data_dirs:
        print("--- Splitting images from " + data_dir)
        index = 0
        for class_name in os.listdir(data_dir):
            if index >= LIMIT != 0:
                return
            index += 1
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                train_class_dir = os.path.join(train_dir, class_name)
                val_class_dir = os.path.join(val_dir, class_name)
                os.makedirs(train_class_dir, exist_ok=True)
                os.makedirs(val_class_dir, exist_ok=True)
                images = os.listdir(class_path)
                random.shuffle(images)
                split_index = int(len(images) * (1 - val_ratio))
                if small_data and not is_custom_images:
                    split_index = int(split_index * SMALL_RATIO)
                train_images = images[:split_index]
                val_images = images[split_index:]
                if small_data and not is_custom_images:
                    val_images = images[split_index:int(split_index / 0.8)]
                for img in train_images:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(train_class_dir, img)
                    shutil.copyfile(src, dst)
                for img in val_images:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(val_class_dir, img)
                    shutil.copyfile(src, dst)
                print(f"Class '{class_name}' split into {len(train_images)} training and {len(val_images)} validation images.")

        is_custom_images = True


def load_dataset(data_dir, img_size=(64, 64)):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_path.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        width, height = img.size
                        min_dim = min(width, height)
                        left = (width - min_dim) / 2
                        top = (height - min_dim) / 2
                        right = (width + min_dim) / 2
                        bottom = (height + min_dim) / 2
                        img = img.crop((left, top, right, bottom))
                        img = img.resize(img_size)
                        img = np.array(img, dtype=np.float32)
                        images.append(img)
                        labels.append(label)
                    except OSError as e:
                        print(f"Skipping corrupted image: {img_path} - {e}")
    return images, labels, class_names
