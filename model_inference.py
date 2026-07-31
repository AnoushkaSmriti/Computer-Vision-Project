# import joblib
# from skimage.feature import hog
# import cv2
# import numpy as np

# # Load the trained model and label encoder
# classifier = joblib.load('asl_svm_model.pkl')
# label_encoder = joblib.load('label_encoder.pkl')

# # Predict the ASL letter/number from an image
# def predict_asl(image):
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image_resized = cv2.resize(image_gray, (64, 64))
#     features = hog(image_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
#     features = features.reshape(1, -1)

#     prediction = classifier.predict(features)
#     return label_encoder.inverse_transform(prediction)[0]

# # Function to predict gesture from static image path
# def predict_from_image_path(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Cannot load image {image_path}")
#         return None
#     predicted_label = predict_asl(image)
#     return predicted_label


# import cv2
# import os
# from skimage.feature import hog
# import numpy as np
# import joblib

# # Load the trained model and label encoder
# classifier = joblib.load('asl_svm_model.pkl')
# label_encoder = joblib.load('label_encoder.pkl')

# # Predict the ASL letter/number from an image
# def predict_asl(image):
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image_resized = cv2.resize(image_gray, (64, 64))
#     features = hog(image_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
#     features = features.reshape(1, -1)

#     prediction = classifier.predict(features)
#     return label_encoder.inverse_transform(prediction)[0]

# # Function to predict gesture from static image path
# def predict_from_image_path(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Cannot load image {image_path}")
#         return None
#     predicted_label = predict_asl(image)
#     return predicted_label

# # Display each image in a folder with the predicted ASL letter/number
# def predict_and_display_images(image_folder):
#     if not os.path.isdir(image_folder):
#         print("Error: Specified dataset path does not exist.")
#         return

#     print("Processing images in folder:", image_folder)
#     for label in os.listdir(image_folder):
#         label_dir = os.path.join(image_folder, label)
        
#         # Check if this is a directory containing images
#         if os.path.isdir(label_dir):
#             print(f"Processing label folder: {label}")
#             for image_file in os.listdir(label_dir):
#                 image_path = os.path.join(label_dir, image_file)
#                 print(f"Processing image: {image_file}")
                
#                 predicted_label = predict_from_image_path(image_path)
                
#                 if predicted_label:
#                     print(f"Predicted Label: {predicted_label} for image: {image_file}")
                    
#                     image = cv2.imread(image_path)
#                     if image is not None:
#                         # Add prediction text to the image
#                         cv2.putText(image, f'Prediction: {predicted_label}', (10, 30), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#                         cv2.imshow(f"Predicted: {predicted_label}", image)

#                         # Wait for a key press; press 'q' to quit
#                         key = cv2.waitKey(0)
#                         if key == ord('q'):
#                             print("Exiting the image display.")
#                             cv2.destroyAllWindows()
#                             return
#                     else:
#                         print(f"Error: Cannot load image {image_file}")
#                 else:
#                     print(f"Skipping: Prediction failed for {image_file}")
    
#     cv2.destroyAllWindows()

# # Run the function
# DATASET_PATH = r'D:\\PROJECT\\COMPUTER_VISION_SIGNLANG_DETECTION\\new_images'
# predict_and_display_images(DATASET_PATH)


import joblib
from skimage.feature import hog
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Enable interactive mode
plt.ion()

# Load the trained model and label encoder
classifier = joblib.load('asl_svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Predict the ASL letter/number from an image
def predict_asl(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, (64, 64))
    features = hog(image_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    features = features.reshape(1, -1)

    prediction = classifier.predict(features)
    return label_encoder.inverse_transform(prediction)[0]

# Function to predict gesture from static image path
def predict_from_image_path(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return None, None
    predicted_label = predict_asl(image)
    return predicted_label, image

# Display each image in a separate window with the predicted ASL letter/number using matplotlib
def predict_and_display_images(image_folder):
    if not os.path.isdir(image_folder):
        print("Error: Specified dataset path does not exist.")
        return

    print("Processing images in folder:", image_folder)
    for label in os.listdir(image_folder):
        label_dir = os.path.join(image_folder, label)
        
        # Check if this is a directory containing images
        if os.path.isdir(label_dir):
            print(f"Processing label folder: {label}")
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                print(f"Processing image: {image_file}")
                
                predicted_label, image = predict_from_image_path(image_path)
                
                if predicted_label:
                    print(f"Predicted Label: {predicted_label} for image: {image_file}")
                    
                    # Convert image to RGB for matplotlib
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Display the image in a new figure window with the predicted label
                    plt.figure()
                    plt.imshow(image_rgb)
                    plt.title(f'Prediction: {predicted_label}')
                    plt.axis('off')  # Hide axis for a cleaner look
                    plt.pause(0.5)  # Small pause to ensure the image loads properly

                else:
                    print(f"Skipping: Prediction failed for {image_file}")

    plt.ioff()  # Turn off interactive mode after all images are processed

# Run the function
DATASET_PATH = r'D:\\PROJECT\\COMPUTER_VISION_SIGNLANG_DETECTION\\new_images'
predict_and_display_images(DATASET_PATH)

