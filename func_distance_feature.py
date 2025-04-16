import math
import numpy as np

def calculate_euclidean_distance(a, b):
    return math.sqrt(
        (b['x'] - a['x'])**2 +
        (b['y'] - a['y'])**2 +
        (b['z'] - a['z'])**2
    )

def calculate_angle(a, b, c):
    # Calculate the angle at point b formed by points a, b, c
    ab = np.array([a['x'] - b['x'], a['y'] - b['y'], a['z'] - b['z']])
    cb = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])
    dot_product = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    angle_rad = math.acos(dot_product / (norm_ab * norm_cb + 1e-6))  # Prevent division by zero
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def calculate_head_to_feet_distance(landmarks):
    distances = {}
    
    head_landmarks = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
    feet_landmarks = ['left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']
    
    # Misalnya, gunakan 'nose' sebagai titik referensi kepala
    head_point = landmarks.get('nose', {'x': 0.0, 'y': 0.0, 'z': 0.0})
    
    for foot in feet_landmarks:
        distance = calculate_euclidean_distance(head_point, landmarks.get(foot, {'x': 0.0, 'y': 0.0, 'z': 0.0}))
        distances[f'dist_nose_{foot}'] = distance
    
    # Anda bisa menambahkan rata-rata atau total jarak jika diperlukan
    distances['avg_dist_head_to_feet'] = np.mean(list(distances.values()))
    
    return distances

def extract_engineered_features(landmarks):
    features = {}
    # Example: Distance between left_shoulder and left_elbow
    features['dist_left_shoulder_elbow'] = calculate_euclidean_distance(
        landmarks['left_shoulder'], landmarks['left_elbow']
    )
    
    # Example: Angle at left_elbow
    features['angle_left_elbow'] = calculate_angle(
        landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']
    )
    
    # Example: Distance between right_shoulder and right_elbow
    features['dist_right_shoulder_elbow'] = calculate_euclidean_distance(
        landmarks['right_shoulder'], landmarks['right_elbow']
    )
    
    # Example: Angle at right_elbow
    features['angle_right_elbow'] = calculate_angle(
        landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']
    )
    # Fitur baru: Jarak Kepala ke Kaki
    head_to_feet_distances = calculate_head_to_feet_distance(landmarks)
    features.update(head_to_feet_distances)
    
    # Add more engineered features as needed
    return features