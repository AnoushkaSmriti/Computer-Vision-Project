import cv2 
from hand_detection import detect_hand, find_hand_contour
from model_inference import predict_asl

def process_video():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = detect_hand(frame)
        hand_contour, segmented_hand = find_hand_contour(mask, frame)

        if hand_contour is not None:
            predicted_label = predict_asl(segmented_hand)
            cv2.putText(segmented_hand, f'Predicted: {predicted_label}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('ASL Hand Detection', segmented_hand)

        cv2.imshow('Mask', mask)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()