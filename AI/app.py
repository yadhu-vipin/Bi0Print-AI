import os
import requests
from flask import Flask, request, render_template, jsonify, session
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session handling

MODEL_PATH = "bloodtype_model.keras"

# Google Drive Direct Download Link (Modify with Your File ID)
GDRIVE_FILE_ID = "1i7OPo_14H0htgO_roKpPX8MmRU_GKZ9m"
GDRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

def download_model():
    """Download the model if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        response = requests.get(GDRIVE_DOWNLOAD_URL, stream=True)
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print("Model download complete!")

# Download the model before loading it
download_model()
model = tf.keras.models.load_model(MODEL_PATH)

img_size = (128, 128)
class_labels = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded."})

    files = request.files.getlist('files')
    predictions = []

    for file in files:
        if file.filename == '':
            continue  # Skip empty files

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            predicted_class = class_labels[class_index]

            predictions.append({"file_path": file_path, "prediction": predicted_class})

        except Exception as e:
            predictions.append({"file_path": file_path, "error": f"Prediction failed: {str(e)}"})

    session["predictions"] = predictions  # Store predictions in session
    return jsonify({"success": True})  # Respond without long URLs

@app.route('/result')
def result():
    predictions = session.get("predictions", [])  # Retrieve predictions from session
    return render_template("result.html", predictions=predictions)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT isn't set
    app.run(host='0.0.0.0', port=port)