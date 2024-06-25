# Computer Vision Project
Computer vision project with the purpose of classifying Pokémon images

## Overview

The main aim of the project is to utilize computer vision techniques to train a model so that,
given an image representing a Pokémon, it can tell which Pokémon it is. The model can be
trained using images from video games, animations, drawings, or even physical objects, giving
us a wide variety of data. This project is relevant as it demonstrates the practical application of
image classification in the context of popular culture and gaming. This project could, for
example, be used to build a larger artificial intelligence that could play Pokémon games on its
own. Our project could recognize Pokémon in the wild, and another might use this data to build
a team that could defeat it, for example.

## Installation

1. Get the dataset:
   1. Download the dataset from kaggle by following this link: https://www.kaggle.com/datasets/echometerhhwl/pokemon-gen-1-38914
   2. Extract the folder in the project root (the directory must be called 'data')
2. Get the custom images:
   1. Download the added images by following this link: https://drive.google.com/file/d/10YbsdKGSgny1GKtRPIWGWaibJw-kA6b6/view?usp=sharing
   2. Extract the folder in the project root (the directory must be called 'CustomImages')

The project tree should look like this:

```shell
.
├── CustomImages
│     ├── ClassName01
│     └── ...
├── data
│     ├── ClassName01
│     └── ...
├── models
│     ├── Model1.pth
│     └── ...
├── plots
│     ├── Plot1.png
│     └── ...
└── src
      ├── main.py
      └── ...
```

## Execution

### Training

Train models and display training statistics

```shell
python main.py [options]
```

The `options` are the following:
- `-r`, `--reload`: Re-splits data, re-creates data cache and re-trains the models
- `-s`, `--small`: Use of a smaller dataset (for testing purposes)
- `-h`, `--help`: Displays this help

### Testing

Use the trained model to predict the classes of input images

```shell
python pred.py <directory_name>
```

The `directory_name` is the name of the directory where the images you want to classify are located.
This directory must be at the project root, and directly contain the images you want to classify.