import dlib
import numpy as np
import os
import pickle

# Load Dlib's face detection and recognition models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Function to get face encodings from known images
def get_face_encodings(image_folder):
    known_face_encodings = []
    known_face_names = []

    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(image_folder, image_name)
            face_image = dlib.load_rgb_image(image_path)  # Load image using dlib
            detected_faces = detector(face_image)
            for face in detected_faces:
                shape = shape_predictor(face_image, face)
                face_encoding = np.array(facerec.compute_face_descriptor(face_image, shape))
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(image_name)[0])
                break  # Assuming only one face per image

    return known_face_encodings, known_face_names

# Change the path to your images folder
images_folder = 'training'

# Save face encodings and names
known_face_encodings, known_face_names = get_face_encodings(images_folder)
with open('known_faces.pkl', 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)
