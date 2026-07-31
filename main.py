import cv2
import os
from model_inference import predict_from_image_path
from video_capture import process_video

# Define the dataset path directly
DATASET_PATH = r'D:\\PROJECT\\COMPUTER_VISION_SIGNLANG_DETECTION\\new_images'

# Function to predict on each image in the specified folder and display results
def predict_on_images(image_folder):
    if not os.path.isdir(image_folder):
        print("Error: Specified dataset path does not exist.")
        return
    
    print("Dataset path exists. Processing images...")
    for label in os.listdir(image_folder):
        label_dir = os.path.join(image_folder, label)
        
        # Check if this is a directory
        if os.path.isdir(label_dir):
            print(f"Processing label folder: {label}")
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                print(f"Attempting to load image: {image_file}")
                
                predicted_label = predict_from_image_path(image_path)
                
                if predicted_label:
                    image = cv2.imread(image_path)
                    if image is not None:
                        cv2.putText(image, f'Prediction: {predicted_label}', (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.imshow(f"Predicted: {predicted_label}", image)

                        key = cv2.waitKey(0)
                        if key == ord('q'):
                            print("Exiting the image processing.")
                            cv2.destroyAllWindows()
                            return
                    else:
                        print(f"Error: Cannot load image {image_file}")
                else:
                    print(f"Skipping: Prediction failed for {image_file}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Choose mode: 'image' for static images, 'video' for real-time video: ").strip().lower()
    
    if mode == 'image':
        predict_on_images(DATASET_PATH)
    elif mode == 'video':
        process_video()
    else:
        print("Invalid mode. Choose 'image' or 'video'.")
