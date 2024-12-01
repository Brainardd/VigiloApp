import cv2
import dlib
import math
import time
import csv
from collections import deque
from scipy.spatial import distance as dist
import mediapipe as mp
import numpy as np
import os

# EAR Calculation Function
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# MOR Calculation Function
def calculate_mor(landmarks):
    vertical = dist.euclidean(landmarks[51], landmarks[57])  # Vertical distance
    horizontal = dist.euclidean(landmarks[48], landmarks[54])  # Horizontal distance
    mor = vertical / horizontal if horizontal != 0 else 0
    return mor

# Head Tilt Calculation Function
def calculate_tilt_angle(landmarks):
    nose = landmarks[30]  # Nose tip
    chin = landmarks[8]   # Chin
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

# Check if hand is covering the mouth
def is_mouth_covered(landmarks, hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    # Define mouth bounding box based on landmarks
    mouth_x_min = min([landmarks[i][0] for i in range(48, 54)])
    mouth_x_max = max([landmarks[i][0] for i in range(48, 54)])
    mouth_y_min = min([landmarks[i][1] for i in range(51, 57)])
    mouth_y_max = max([landmarks[i][1] for i in range(51, 57)])

    mouth_box = (mouth_x_min, mouth_y_min, mouth_x_max, mouth_y_max)

    # Iterate through hand landmarks and check overlap
    for hand_landmark in hand_landmarks:
        hand_points = [
            (int(hand_landmark.landmark[i].x * w), int(hand_landmark.landmark[i].y * h))
            for i in range(21)
        ]

        # Calculate hand bounding box
        hand_x_min = min([point[0] for point in hand_points])
        hand_x_max = max([point[0] for point in hand_points])
        hand_y_min = min([point[1] for point in hand_points])
        hand_y_max = max([point[1] for point in hand_points])

        hand_box = (hand_x_min, hand_y_min, hand_x_max, hand_y_max)

        # Check for bounding box overlap
        if (
            hand_box[0] < mouth_box[2]
            and hand_box[2] > mouth_box[0]
            and hand_box[1] < mouth_box[3]
            and hand_box[3] > mouth_box[1]
        ):
            return True
    return False

# Mediapipe Hands Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/brain/Desktop/VIGILO_FATIGUE_DETECTION/Training/EAR & PERCLOS Detection/shape_predictor_68_face_landmarks.dat")

# Fatigue Parameters
EAR_THRESHOLD = 0.23
MOR_THRESHOLD = 0.45
PERCLOS_THRESHOLD = 50  # Percentage
FOM_THRESHOLD = 3       # Open mouth events per minute
TILT_THRESHOLD = 10     # Degrees
FATIGUE_DURATION = 3    # Seconds to trigger fatigue warning
TIME_WINDOW = 30        # Sliding window size (seconds)
FPS = 15                # Approximate frames per second
NO_FACE_DETECTED_THRESHOLD = 3  # Seconds before flagging "No Face Detected"

# Sliding windows for metrics
closed_frames = deque(maxlen=TIME_WINDOW * FPS)
mouth_open_counts = deque(maxlen=TIME_WINDOW * FPS)
tilt_durations = deque(maxlen=TIME_WINDOW * FPS)
fatigue_start_time = None
last_face_detected_time = time.time()  # Initialize to current time

# Initialize CSV File
csv_file_path = "fatigue_metrics.csv"
file_exists = os.path.isfile(csv_file_path)

csv_file = open(csv_file_path, "a", newline="")
csv_writer = csv.writer(csv_file)

if not file_exists:
    csv_writer.writerow(["Timestamp", "EAR", "PERCLOS", "MOR", "FOM", "Tilt Angle", "Fatigue Detected"])

# Timer for logging per second
last_logged_time = time.time()

# Capture video from webcam
cap = cv2.VideoCapture(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        faces = detector(gray)

        fatigue_status = "No"
        hand_covering_mouth = False
        face_detected = len(faces) > 0  # Check if any face is detected

        if face_detected:
            last_face_detected_time = time.time()  # Update the last detected time

        # Detect hands covering mouth if face is detected
        if face_detected and results.multi_hand_landmarks:
            for face in faces:
                shape = predictor(gray, face)
                landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                hand_covering_mouth = is_mouth_covered(landmarks, results.multi_hand_landmarks, frame.shape)

        for face in faces:
            shape = predictor(gray, face)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            # EAR Calculation
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            closed = ear < EAR_THRESHOLD
            closed_frames.append(closed)

            # PERCLOS Calculation
            perclos = sum(closed_frames) / len(closed_frames) * 100 if closed_frames else 0

            # MOR Calculation
            if not hand_covering_mouth:
                mor = calculate_mor(landmarks)
            else:
                mor = mor if 'mor' in locals() else 0  # Freeze MOR when mouth is covered

            mouth_open = mor > MOR_THRESHOLD
            mouth_open_counts.append(mouth_open)

            # FOM Calculation
            fom = sum(mouth_open_counts)  # Frequency of open mouth events

            # Head Tilt Calculation
            tilt_angle = calculate_tilt_angle(landmarks)
            tilt_detected = abs(tilt_angle) > TILT_THRESHOLD
            tilt_durations.append(tilt_detected)

            # Sustained Tilt Duration
            sustained_tilt_duration = sum(tilt_durations) / FPS

            # Combine Fatigue Conditions
            if (perclos > PERCLOS_THRESHOLD and
                fom > FOM_THRESHOLD and
                sustained_tilt_duration > FATIGUE_DURATION and
                mouth_open and
                closed):
                fatigue_status = "Yes"

            # Visualize Metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.2f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"MOR: {mor:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"FOM: {fom} events", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Tilt: {tilt_angle:.2f} deg", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw Landmarks
            for (x, y) in left_eye + right_eye + landmarks[48:60]:  # Eyes and mouth
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            cv2.circle(frame, landmarks[30], 5, (0, 255, 0), -1)    # Nose tip
            cv2.circle(frame, landmarks[8], 5, (0, 255, 0), -1)     # Chin

        # Check for "No Face Detected"
        current_time = time.time()
        if (current_time - last_face_detected_time) > NO_FACE_DETECTED_THRESHOLD:
            fatigue_status = "No Face Detected"
            cv2.putText(frame, "NO FACE DETECTED!", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


        # Log Metrics to CSV
        if current_time - last_logged_time >= 1:  # Log every second
            csv_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
                ear if face_detected else "N/A",
                perclos if face_detected else "N/A",
                mor if face_detected else "N/A",
                fom if face_detected else "N/A",
                tilt_angle if face_detected else "N/A",
                fatigue_status
            ])
            last_logged_time = current_time

        # Show frame
        cv2.imshow("Frame", frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    csv_file.close()  # Ensure the file is closed
    cap.release()
    cv2.destroyAllWindows()
