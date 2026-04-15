from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import base64

app = Flask(__name__)
CORS(app)

# Load model and actions
DATA_PATH = os.path.join("C:\\Users\\IZISS\\Desktop\\Sign-Bridge\\Data_Set")
actions = np.array([f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))])

# Build model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'))
model.add(LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('action.h5')

# MediaPipe global - ek baar hi initialize hoga
mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

sequence_buffer = []

def mediapipe_detection(image, model_mp):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model_mp.process(image)
    image.flags.writeable = True
    return results

def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Yeh line HATAO upar se (global holistic)
# holistic = mp_holistic.Holistic(...)

# /predict route ko yeh banao:
@app.route('/predict', methods=['POST'])
def predict():
    global sequence_buffer
    try:
        data = request.json
        img_data = base64.b64decode(data['frame'])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=True  # ← Yeh add karo! Timestamp issue fix hoga
        ) as holistic:
            results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)

        sequence_buffer.append(keypoints)
        sequence_buffer = sequence_buffer[-30:]

        if len(sequence_buffer) == 30:
            res = model.predict(np.expand_dims(sequence_buffer, axis=0))[0]
            confidence = float(np.max(res))
            predicted = actions[np.argmax(res)]
            if confidence > 0.7:
                return jsonify({'sign': predicted, 'confidence': confidence})

        return jsonify({'sign': '', 'confidence': 0.0})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    global sequence_buffer
    sequence_buffer = []
    return jsonify({'status': 'reset'})

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'actions': actions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)