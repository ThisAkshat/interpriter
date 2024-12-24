import cv2
import numpy as np
import os
import random
import mediapipe as mp
import pyttsx3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# Initialize text-to-speech engine
engine = pyttsx3.init()

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

# Adjusted Augmentation Functions

def random_flip_keypoints(keypoints, flip_prob=0.3):
    """Flip the keypoints horizontally with a given probability."""
    if random.random() < flip_prob:
        # Flip the x-coordinates of the keypoints (assuming x is the 0th dimension)
        keypoints[::3] = -keypoints[::3]
    return keypoints

def random_scale_keypoints(keypoints, scale_range=(0.9, 1.1)):
    """Randomly scale the keypoints."""
    scale_factor = random.uniform(scale_range[0], scale_range[1])
    keypoints[::3] *= scale_factor  # Apply scaling to x-coordinates
    keypoints[1::3] *= scale_factor  # Apply scaling to y-coordinates
    return keypoints

def random_translate_keypoints(keypoints, max_translation=0.05):
    """Randomly translate the keypoints by a small amount."""
    translation_x = random.uniform(-max_translation, max_translation)
    translation_y = random.uniform(-max_translation, max_translation)
    
    keypoints[::3] += translation_x  # Translate x-coordinates
    keypoints[1::3] += translation_y  # Translate y-coordinates
    return keypoints

def augment_keypoints(keypoints):
    """Apply all augmentations to the keypoints."""
    keypoints = random_flip_keypoints(keypoints)
    keypoints = random_scale_keypoints(keypoints)
    keypoints = random_translate_keypoints(keypoints)
    return keypoints


# Function to check if accuracy is 100%
def is_accuracy_100(model, X_test, y_test):
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    accuracy = accuracy_score(ytrue, yhat)
    print(f"Current accuracy: {accuracy * 100:.2f}%")
    return accuracy == 1.0

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
MODEL_FILE = 'action.h5'
# Actions that we try to detect
actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 0  # Start folder from 0

# Ensure base directory exists
os.makedirs(DATA_PATH, exist_ok=True)

label_map = {label: num for num, label in enumerate(actions)}

# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'))
model.add(LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Load weights if available
if os.path.exists(MODEL_FILE):
    model.load_weights(MODEL_FILE)
    print("Loaded model weights from file.")
else:
    print("No pre-trained weights found. Training model from scratch.")

# Check for new data
def check_for_new_data(actions, DATA_PATH):
    current_actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])
    new_actions = [action for action in current_actions if action not in actions]
    incomplete_sequences = []

    for action in current_actions:
        action_path = os.path.join(DATA_PATH, action)
        sequences = [int(seq) for seq in os.listdir(action_path) if seq.isdigit()]
        for seq in sequences:
            seq_path = os.path.join(action_path, str(seq))
            if len(os.listdir(seq_path)) < sequence_length:
                incomplete_sequences.append((action, seq))

    return new_actions, incomplete_sequences

# Prepare the data with augmentations
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            # Apply augmentation to keypoints
            augmented_res = augment_keypoints(res)
            window.append(augmented_res)
        sequences.append(window)
        labels.append(label_map[action])

# Check if data loaded correctly
print(f"Number of sequences: {len(sequences)}")
print(f"Number of labels: {len(labels)}")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Retrain the model until accuracy reaches 100%
while not is_accuracy_100(model, X_test, y_test):
    print("Retraining the model...")
    model.fit(X_train, y_train, epochs=10, callbacks=[TensorBoard(log_dir=log_dir)])  # Shorter epochs for efficiency
    model.save(MODEL_FILE)
    print("Updated model weights saved.")

# Evaluate model performance
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print("Multilabel confusion matrix:")
print(multilabel_confusion_matrix(ytrue, yhat))

print("Accuracy Score:")
print(accuracy_score(ytrue, yhat))

# Visualization for predictions
from scipy import stats

# Generate random colors for the actions
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(actions))]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 10 + num * 40), (int(prob * 100), 40 + num * 40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, actions[num], (0, 35 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Real-time detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.7

cap = cv2.VideoCapture(1)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        # Apply augmentations during real-time prediction
        augmented_keypoints = augment_keypoints(keypoints)
        sequence.append(augmented_keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # Viz logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            engine.say(actions[np.argmax(res)])
                            engine.runAndWait()
                    else:
                        sentence.append(actions[np.argmax(res)])
                        engine.say(actions[np.argmax(res)])
                        engine.runAndWait()

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualize probabilities
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 440), (640, 480), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        cv2.namedWindow('OpenCV Feed', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('OpenCV Feed', cv2.WND_PROP_TOPMOST, 1)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
