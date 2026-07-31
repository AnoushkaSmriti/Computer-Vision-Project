import cv2
import joblib
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

# Load the trained model and label encoder
classifier = joblib.load('asl_svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to preprocess real-time frames (similar to the training preprocessing)
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))  # Ensure consistent size
    
    # Visualize the resized frame for debugging
    cv2.imshow('Resized Frame', resized_frame)
    
    # Compute HOG features
    features, hog_image = hog(resized_frame, orientations=9, pixels_per_cell=(8, 8), 
                              cells_per_block=(2, 2), visualize=True)
    
    # Visualize the HOG image for debugging
    cv2.imshow('HOG Features', hog_image.astype('uint8'))

    return features

# Function for color filtering to detect hand regions
def hand_color_segmentation(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 20], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Mask skin color
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    
    # Apply the mask to extract skin-colored regions
    hand_segment = cv2.bitwise_and(frame, frame, mask=mask)
    
    return hand_segment

# Function for detecting hand contours
def detect_hand_contours(frame):
    # Convert the frame to grayscale and blur it
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Adaptive thresholding for better lighting conditions
    thresholded_frame = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    
    return frame

# Open webcam for real-time video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Define a Region of Interest (ROI) for hand detection to focus on hands only
    roi = frame[100:400, 100:400]  # Adjust the values based on hand positioning
    
    # Detect hand region using color filtering
    hand_segment = hand_color_segmentation(roi)
    
    # Detect contours in the hand-segmented image
    hand_contour_frame = detect_hand_contours(hand_segment.copy())
    
    # Preprocess the current frame (ROI) for prediction
    features = preprocess_frame(roi)
    
    # Predict the ASL sign using the trained classifier
    predicted_label = classifier.predict([features])[0]
    predicted_sign = label_encoder.inverse_transform([predicted_label])[0]
    
    # Display the predicted ASL sign on the video frame
    cv2.putText(hand_contour_frame, f'Predicted: {predicted_sign}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the frame with contours
    cv2.imshow('ASL Detection with Hand Segmentation', hand_contour_frame)
    
    # Show the ROI frame (for debugging)
    cv2.imshow('ROI', roi)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()