# Garbage Detection Using CNN

## Overview
This project aims to develop a garbage detection system using Convolutional Neural Networks (CNN) to classify images captured by a camera into different categories of garbage. The system will help in identifying and categorizing waste in real-time, promoting better waste management practices.

## Features
- Real-time garbage classification using camera input.
- Ability to classify multiple types of garbage.
- User-friendly interface for displaying results.
- Supports a custom dataset for training.

## Technologies Used
- Python
- TensorFlow/Keras for building and training the CNN model
- OpenCV for camera access and image processing
- PIL (Python Imaging Library) for image manipulation

## Dataset
The dataset used for this project should contain labeled images of various types of garbage. Make sure to organize your dataset in the following structure:
garbage-detection/
│
├── train_data/                  # Directory for training images
│   ├── class_1/                 # Class 1 images (e.g., plastic)
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_2/                 # Class 2 images (e.g., paper)
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── class_n/                 # Class n images (e.g., metal)
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── models/                      # Directory for saving trained models
│   ├── model.h5                 # Example model file
│   └── ...
│
├── scripts/                     # Directory for scripts
│   ├── main.py                  # Main script for running the application
│   ├── train.py                 # Script for training the model
│   └── utils.py                 # Utility functions (optional)
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License information
