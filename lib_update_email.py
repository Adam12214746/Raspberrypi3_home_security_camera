import os
import cv2
import numpy as np
import dlib
import pickle
import time
import threading
import smtplib
from email.message import EmailMessage
import imghdr
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants for email and face detection
EMAIL_COOLDOWN = 15  # Email cooldown in seconds
UNKNOWN_DETECTION_WINDOW = 5  # Time window to detect unknown faces
UNKNOWN_DETECTION_COUNT = 4  # Number of unknown faces to detect within the window
FACE_THRESHOLD = 0.4  # Face recognition threshold for identifying known faces

# Email credentials with default values if not set in the environment
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL', "uzmir6358@gmail.com")


# Load pre-existing encodings
def load_face_encodings(file_path="known_faces.pkl"):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        exit(1)

# Initialize dlib's face detection and recognition models
def initialize_dlib_models():
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    return detector, shape_predictor, facerec

# Function to send email with an attachment
def send_email_with_attachment(subject, body, receiver_email, sender_email, sender_password, attachment):
    message = EmailMessage()
    message.set_content(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = receiver_email

    # Add attachment if provided
    if attachment:
        with open(attachment, 'rb') as f:
            file_data = f.read()
            file_type = imghdr.what(f.name)
            file_name = f.name
        message.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

    # Send email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(message)
            print(f"Email sent to {receiver_email}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Shared state for threading
class SharedState:
    def __init__(self):
        self.last_email_time = 0
        self.unknown_faces_detected = 0
        self.unknown_face_start_time = None
        self.known_faces_count = 0
        self.unknown_faces_count = 0
        self.lock = threading.Lock()

shared_state = SharedState()

# Function to process each frame for face detection and recognition
def process_frame(frame, known_face_encodings, known_face_names, detector, shape_predictor, facerec):
    with shared_state.lock:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_frame)
        
        for face in faces:
            shape = shape_predictor(rgb_frame, face)
            face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))

            # Compare face encodings
            distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
            best_match_index = np.argmin(distances)
            name = "Unknown"
            
            if distances[best_match_index] < FACE_THRESHOLD:  # Threshold for recognizing a face
                name = known_face_names[best_match_index]
                shared_state.known_faces_count += 1
            else:
                shared_state.unknown_faces_count += 1
                if shared_state.unknown_face_start_time is None:
                    shared_state.unknown_face_start_time = time.time()
                    shared_state.unknown_faces_detected = 1
                else:
                    shared_state.unknown_faces_detected += 1
                    elapsed_time = time.time() - shared_state.unknown_face_start_time
                    if elapsed_time > UNKNOWN_DETECTION_WINDOW:
                        if shared_state.unknown_faces_detected >= UNKNOWN_DETECTION_COUNT:
                            # Take a snapshot and send an email
                            img_name = f"unknown_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(img_name, frame)
                            send_email_with_attachment(
                                "Unrecognized Person Detected",
                                "Several unknown people were detected.",
                                RECEIVER_EMAIL, SENDER_EMAIL, SENDER_PASSWORD,
                                img_name
                            )
                            shared_state.last_email_time = time.time()
                        # Reset the detection window
                        shared_state.unknown_face_start_time = None
                        shared_state.unknown_faces_detected = 0

            # Draw bounding box around face
            (top, right, bottom, left) = (face.top(), face.right(), face.bottom(), face.left())
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Main function to capture video and process frames
def main():
    # Initialize models and load encodings
    detector, shape_predictor, facerec = initialize_dlib_models()
    known_face_encodings, known_face_names = load_face_encodings()

    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Process frames
    start_time = time.time()
    total_frames_processed = 0
    program_duration = 20  # Duration for which the program should run, in seconds
    
    while time.time() - start_time < program_duration:
        ret, frame = video_capture.read()
        if not ret:
            break

        total_frames_processed += 1
        frame = process_frame(frame, known_face_encodings, known_face_names, detector, shape_predictor, facerec)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and display stats
    video_capture.release()
    cv2.destroyAllWindows()

    total_duration = time.time() - start_time
    fps = total_frames_processed / total_duration if total_duration > 0 else 0

    print(f"Total known face detections: {shared_state.known_faces_count}")
    print(f"Total unknown face detections: {shared_state.unknown_faces_count}")
    print(f"Total face detections: {total_frames_processed}")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print(f"Detection speed: {fps:.2f} frames per second")
    print(f"Average faces detected per frame: {total_frames_processed / total_duration if total_duration > 0 else 0:.2f}")
    print(f"Total frames processed: {total_frames_processed}")

if __name__ == "__main__":
    main()
