import os
import cv2
import numpy as np
from skimage.feature import hog

# Preprocessing function to load, resize, and extract HOG features from images
def preprocess_image(image_path):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    image = cv2.resize(image, (64, 64))
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

# Function to load dataset from the given path
def load_dataset(dataset_path):
    print(dataset_path)
    X = []
    y = []
    for label in os.listdir(dataset_path):
        label_dir = os.path.join(dataset_path, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                features = preprocess_image(image_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

load_dataset(r'D:\\PROJECT\\COMPUTER_VISION_SIGNLANG_DETECTION\\data\\asl_dataset')