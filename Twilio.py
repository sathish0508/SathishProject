from flask import Flask, request, jsonify
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import euclidean
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Paths for training images (relative paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
IMAGE_DIR = os.path.join(BASE_DIR, "images")

# Update paths to be relative
image_paths = [
    os.path.join(IMAGE_DIR, "Eyeswithglucoma.jpg"),  # Glaucoma image
    os.path.join(IMAGE_DIR, "Sampb.jpg"),           # Glaucoma image
    os.path.join(IMAGE_DIR, "normalEsys2.jpg"),     # No Glaucoma image
    os.path.join(IMAGE_DIR, "NormalEys.jpg"),       # No Glaucoma image
]

# Labels for the images (binary: Glaucoma vs No Glaucoma)
labels = ['Glaucoma', 'Glaucoma', 'No Glaucoma', 'No Glaucoma']

# Resize images to 64x64 for consistency
image_size = (64, 64)
data = []
target = []

# Preprocess the images (resize and flatten them)
for image_path, label in zip(image_paths, labels):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue
    image = cv2.resize(image, image_size)  # Resize image
    data.append(image)
    target.append(label)

# Convert data and target to numpy arrays
data = np.array(data)
target = np.array(target)

# Flatten the images to 1D arrays
data = data.reshape(data.shape[0], -1)

# Encode the labels (binary: Glaucoma vs No Glaucoma)
le = LabelEncoder()
target = le.fit_transform(target)  # 0: No Glaucoma, 1: Glaucoma

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)

# Train an SVM classifier with probability estimation
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Helper function to calculate affected percentage
def calculate_affected_percentage(uploaded_image_features):
    # Reference glaucoma images (using relative paths)
    glaucoma_images = [
        os.path.join(IMAGE_DIR, "Eyeswithglucoma.jpg"),
        os.path.join(IMAGE_DIR, "Sampb.jpg"),
    ]

    distances = []
    for path in glaucoma_images:
        ref_image = cv2.imread(path)
        if ref_image is None:
            continue
        ref_image_resized = cv2.resize(ref_image, image_size)
        ref_image_features = ref_image_resized.flatten()

        # Calculate Euclidean distance between the uploaded image and the reference image
        dist = euclidean(uploaded_image_features, ref_image_features)
        distances.append(dist)

    # Normalize the distance to calculate the affected percentage
    max_distance = max(distances)
    min_distance = min(distances)
    diff = distances[0]  # Distance between uploaded image and the first reference image

    # Calculate the affected percentage
    affected_percentage = ((max_distance - diff) / (max_distance - min_distance)) * 100
    return affected_percentage

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the uploaded image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Unable to process the image"}), 400

        # Process and classify the uploaded sample image
        sample_image_resized = cv2.resize(img, image_size)
        sample_image_resized = sample_image_resized.reshape(1, -1)  # Flatten for prediction

        # Predict the class probabilities for the sample image
        probabilities = model.predict_proba(sample_image_resized)

        # The model's prediction confidence for Glaucoma (class 1)
        predicted_probability = probabilities[0][1] * 100  # in percentage

        # Output the result
        if predicted_probability > 50:
            # Glaucoma detected
            affected_percentage = calculate_affected_percentage(sample_image_resized.flatten())
            severity = "High" if affected_percentage > 70 else "Medium" if affected_percentage > 40 else "Low"
            return jsonify({
                "glaucoma_detected": "Yes",
                "affected_percentage": affected_percentage,
                "confidence": predicted_probability,
                "severity": severity
            })
        else:
            # No Glaucoma detected
            return jsonify({
                "glaucoma_detected": "No",
                "confidence": 100 - predicted_probability,
                "severity": "None"
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
