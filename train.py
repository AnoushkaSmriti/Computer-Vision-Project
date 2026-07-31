import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Set the path to your ASL dataset
DATASET_PATH = r"C:\Users\Reliance Digital\Documents\Computer Vision Project\ASL"

# Preprocessing function to load, resize, and extract HOG features from images
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    image = cv2.resize(image, (64, 64))  # Resize image to 64x64
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features

# Load dataset
X = []
y = []
for label in os.listdir(DATASET_PATH):
    label_dir = os.path.join(DATASET_PATH, label)
    if os.path.isdir(label_dir):
        print(f"Processing label: {label}")
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            features = preprocess_image(image_path)
            if features is not None:  # Avoid invalid images
                X.append(features)
                y.append(label)

X = np.array(X)
y = np.array(y)

# Check if dataset is loaded properly
print(f"Total samples: {len(X)}")
if len(X) == 0:
    raise Exception("Dataset loading failed. Check dataset path or image files.")

# Encode the labels into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and label encoder for later use
joblib.dump(classifier, 'asl_svm_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Model and label encoder saved successfully.")

# Function to detect hand color based on HSV range
def detect_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

# Function to find contours of the detected hand
def find_hand_contour(mask, frame):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
        return max_contour
    return None

# Main function to capture video and detect letters/numbers in real time
def process_video():
    cap = cv2.VideoCapture(0)  # Use camera for real-time input
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect hand based on color range
        mask = detect_hand(frame)
        
        # Find and draw the hand contour on the original frame
        hand_contour = find_hand_contour(mask, frame)

        # If a hand contour is detected, predict gesture
        if hand_contour is not None:
            # Extract features from the hand region for prediction
            x, y, w, h = cv2.boundingRect(hand_contour)
            hand_region = frame[y:y+h, x:x+w]
            hand_region = cv2.resize(hand_region, (64, 64))
            hand_region_gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            features = hog(hand_region_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            features = features.reshape(1, -1)  # Reshape for prediction
            
            # Make a prediction
            if features.shape[1] == X.shape[1]:  # Ensure feature shape matches
                prediction = classifier.predict(features)
                predicted_label = label_encoder.inverse_transform(prediction)
                cv2.putText(frame, f'Predicted: {predicted_label[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the result
        cv2.imshow('ASL Hand Detection', frame)
        cv2.imshow('Mask', mask)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the video processing function
process_video()
