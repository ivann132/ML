import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import model_from_json
import tensorflow as tf


# Load the model architecture and weights
json_file = open('final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('final_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Simpan model TFLite ke file
with open('final_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()


# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
actions = np.array(['A','Absen','akhir','apung','awal','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
# Function to process a single frame's keypoints
def process_keypoints(frame, hands_model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb_frame)
    keypoints = np.zeros((21 * 3,))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, lm in enumerate(hand_landmarks.landmark):
                keypoints[i * 3] = lm.x
                keypoints[i * 3 + 1] = lm.y
                keypoints[i * 3 + 2] = lm.z
        return keypoints, True, results.multi_hand_landmarks
    return keypoints, False, None

cap = cv2.VideoCapture(0)

keypoints_sequence = []  # To store a sequence of 30 frames' keypoints

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Process the current frame to get keypoints and hand landmarks
    keypoints, landmarks_detected, hand_landmarks = process_keypoints(frame, hands)

    # Draw landmarks and bounding box if landmarks are detected
    if landmarks_detected:
        # Draw hand landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Get bounding box coordinates
        bbox_coords = []
        for lm in hand_landmarks[0].landmark:
            h, w, _ = frame.shape
            x, y = int(lm.x * w), int(lm.y * h)
            bbox_coords.append((x, y))

        # Draw bounding box
        bbox_coords = np.array(bbox_coords)
        bbox = cv2.boundingRect(bbox_coords)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

        # Append the processed keypoints to the sequence
        keypoints_sequence.append(keypoints)

        # Once we have a full sequence, make a prediction
        if len(keypoints_sequence) == 30:
            keypoints_data = np.array(keypoints_sequence).reshape(-1, 30, 63, 1).astype(np.float32)
            
            # Prepare input tensor for inference
            input_tensor = interpreter.get_input_details()[0]['index']
            interpreter.set_tensor(input_tensor, keypoints_data)

            # Run inference
            interpreter.invoke()

            # Get output tensor and make prediction
            output_tensor = interpreter.get_output_details()[0]['index']
            prediction = interpreter.get_tensor(output_tensor)

            gesture_id = np.argmax(prediction)

            # Display the prediction on the frame
            gesture_label = actions[gesture_id]
            cv2.putText(frame, f'Gesture Label: {gesture_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Reset the sequence
            keypoints_sequence = []

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()