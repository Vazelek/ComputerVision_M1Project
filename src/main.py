import os
import shutil
import random
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import pickle

SMALL_RATIO = 0.05


def display_help():
    print(">>> python main.py [OPTIONS]\n")
    print("\t-r, --reload\tRecreate split_data directory and data cache")
    print("\t-s,  --small\t\tUse of a smaller data file")
    print("\t-h,  --help\t\t\tDisplay this help")


def split_dataset(data_dir, train_dir, val_dir, val_ratio=0.2, small_data=False):
    """
    Splits the dataset into training and validation sets.

    Parameters:
    - data_dir (str): Path to the source data directory containing class subfolders.
    - train_dir (str): Path to the directory where training data will be stored.
    - val_dir (str): Path to the directory where validation data will be stored.
    - val_ratio (float): Proportion of the data to be used for validation.
    """
    # Ensure the training and validation directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate over each class folder in the source data directory
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            # Create corresponding class directories in train and val directories
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            # List all images in the current class directory
            images = os.listdir(class_path)
            random.shuffle(images)  # Shuffle the images list

            # Determine the split point
            split_index = int(len(images) * (1 - val_ratio))

            if small_data:
                split_index = int(split_index*SMALL_RATIO)

            # Split the images into training and validation sets
            train_images = images[:split_index]
            val_images = images[split_index:]

            if small_data:
                val_images = images[split_index:int(split_index / 0.8)]

            # Move images to the training directory
            for img in train_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(train_class_dir, img)
                shutil.copyfile(src, dst)

            # Move images to the validation directory
            for img in val_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(val_class_dir, img)
                shutil.copyfile(src, dst)

            print(f"Class '{class_name}' split into {len(train_images)} training and {len(val_images)} validation images.")


def load_dataset(data_dir, img_size=(64, 64)):
    """
    Loads the dataset from the specified directory.

    Parameters:
    - data_dir (str): Path to the data directory containing class subfolders.

    Returns:
    - images (list): List of image data.
    - labels (list): List of corresponding labels.
    """
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
                        img = Image.open(img_path).convert('RGBA')  # Ensure image is in RGB format

                        # Crop the center square
                        width, height = img.size
                        min_dim = min(width, height)
                        left = (width - min_dim) / 2
                        top = (height - min_dim) / 2
                        right = (width + min_dim) / 2
                        bottom = (height + min_dim) / 2
                        img = img.crop((left, top, right, bottom))

                        img = img.resize(img_size)  # Resize image to the specified siz

                        img = np.array(img, dtype=np.float32)  # Convert image to numpy array and ensure float32 type
                        images.append(img)
                        labels.append(label)
                    except OSError as e:
                        print(f"Skipping corrupted image: {img_path} - {e}")

    return images, labels, class_names


def apply_model(train_images, train_labels, test_images, test_labels, class_names):
    # Build the model
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 4)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    # Make predictions on the test dataset
    predictions = model.predict(test_images)

    # Plot some test images with their predicted labels
    fig = plt.figure(figsize=(8, 8))

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow(test_images[i])
        predicted_label = np.argmax(predictions[i])
        true_label = test_labels[i]
        ax.set_title("Predicted: {}\nTrue:{}".format(class_names[predicted_label], class_names[true_label]))
        ax.axis('off')
    plt.show()


if __name__ == "__main__":
    data_dir = '../data'
    large_split_data_dir = '../split_data'
    small_split_data_dir = '../small_split_data'
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
        if ((sys.argv[1] == "-r" or sys.argv[1] == "--reload") and (sys.argv[2] == "-s" or sys.argv[2] == "--small")) or ((sys.argv[2] == "-r" or sys.argv[2] == "--reload") and (sys.argv[1] == "-s" or sys.argv[1] == "--small")):
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
        split_dataset(data_dir, train_dir, val_dir, val_ratio=0.2, small_data=True)
    elif not small_data and (reload or not os.path.isdir(large_split_data_dir)):
        print("Splitting data")
        if os.path.isdir(large_split_data_dir):
            shutil.rmtree(large_split_data_dir)
        split_dataset(data_dir, train_dir, val_dir, val_ratio=0.2)

    # Save the processed images and load it
    dataset_cache = 'dataset_cache' + ('_small' if small_data else '') + '.pkl'
    if os.path.exists(dataset_cache) and not reload:
        with open(dataset_cache, 'rb') as f:
            train_images, train_labels, test_images, test_labels, class_names = pickle.load(f)
        print("Loaded dataset from cache.")
    else:
        train_images, train_labels, class_names = load_dataset(train_dir)
        test_images, test_labels, _ = load_dataset(val_dir)
        with open(dataset_cache, 'wb') as f:
            pickle.dump((train_images, train_labels, test_images, test_labels, class_names), f)
        print("Processed and saved dataset to cache.")

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    apply_model(train_images, train_labels, test_images, test_labels, class_names)
