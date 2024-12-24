import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# Initialize Mediapipe modules
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_face_mesh = mp.solutions.face_mesh  # Face mesh module

# Function for Mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB 
    image.flags.writeable = False                   # Make image non-writeable
    results = model.process(image)                  # Make predictions
    image.flags.writeable = True                    # Make image writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results

# Function to draw landmarks
def draw_landmarks(image, results):
    # Draw face, pose, and hand landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)  # Face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)    # Pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Right hand connections

# Function to draw styled landmarks
def draw_styled_landmarks(image, results):
    # Face landmarks
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )
    # Pose landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    # Left hand landmarks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Right hand landmarks
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

# Function to extract keypoints
def extract_keypoints(results):
    # Extract pose, face, left hand, and right hand landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join("D:\\Coding\\Python\\ActionDetectionforSignLanguage\\MP_Data")
os.makedirs(DATA_PATH, exist_ok=True)

# Actions that we try to detect
actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])


# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Main logic for training
while True:
    print("\nOptions:")
    print("t. Train a new word")
    print("q. Exit")
    choice = input("Enter your choice: ")
    if choice == "q":
        print("Exiting the program.")
        break
    elif choice == "t":
        new_word = input("Enter the word you want to train: ").strip()
        action_path = os.path.join(DATA_PATH, new_word)

        if new_word in actions:
            response = input(f"The word '{new_word}' already exists. Do you want to add more data for better accuracy? (yes/no): ").strip().lower()
            if response != "yes":
                continue
            else:
                # Get the existing sequences for the word
                existing_sequences = os.listdir(action_path)
                existing_sequence_numbers = [int(seq) for seq in existing_sequences if seq.isdigit()]
                next_sequence_number = max(existing_sequence_numbers, default=-1) + 1  # Start from the next number
        else:
            actions = np.append(actions, new_word)
            os.makedirs(action_path, exist_ok=True)
            next_sequence_number = 0  # Start from 0 if it's a new word

        cap = cv2.VideoCapture(1)
        quit_training = False  # Flag to monitor quit request

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for sequence in range(next_sequence_number, next_sequence_number + no_sequences):  # Continue from next sequence number
                if quit_training:
                    break  
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame. Exiting...")
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, f'STARTING COLLECTION FOR : {new_word}', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {new_word} Video Number {sequence}', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.namedWindow('OpenCV Feed', cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty('OpenCV Feed', cv2.WND_PROP_TOPMOST, 1)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Collecting frames for {new_word} Video Number {sequence}', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.namedWindow('OpenCV Feed', cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty('OpenCV Feed', cv2.WND_PROP_TOPMOST, 1)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(action_path, str(sequence), str(frame_num))
                    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit signal detected. Exiting training...")
                        quit_training = True
                        break
                        

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Please try again.")