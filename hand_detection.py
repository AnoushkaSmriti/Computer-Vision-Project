import cv2
import numpy as np

def detect_hand(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Perform morphological operations to clean up the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # After generating the mask we apply morphological operations to fiz the rough jagged edges of the countouring done later
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask

def find_hand_contour(mask, frame):
    # Find contours in the mask
    # In find_hand_contour function, apply contour approximation

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find the largest contour based on area
        max_contour = max(contours, key=cv2.contourArea)
        
        # Draw the contour on the frame (optional, for visualization)
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
        
        # Create a mask for the segmented hand
        hand_mask = np.zeros_like(frame)  # Create a black mask with the same dimensions as the frame
        cv2.fillPoly(hand_mask, [max_contour], (255, 255, 255))  # Fill the hand contour with white
        
        # Optional: Combine the original frame with the hand mask to show only the hand
        segmented_hand = cv2.bitwise_and(frame, hand_mask)  # Apply the mask to the frame

        return max_contour, segmented_hand  # Return both the contour and the segmented hand image
    
    return None, None  # Return None if no contour found
