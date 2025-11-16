from flask import Flask, request
from joblib import load
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load trained model
model_bundle = load('savedmodel.pth')
clf = model_bundle['model']

# Home page with upload form
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Olivetti Faces Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f2f2f2;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .card {
                background: white;
                padding: 30px 40px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                text-align: center;
                width: 350px;
            }
            input[type="file"] {
                margin: 10px 0;
                padding: 6px;
                border-radius: 6px;
                width: 100%;
                border: 1px solid #ccc;
            }
            input[type="submit"] {
                background-color: #007BFF;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 15px;
            }
            input[type="submit"]:hover {
                background-color: #0056b3;
            }
            h2 {
                margin-bottom: 15px;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Olivetti Faces Classifier</h2>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br><br>
                <input type="submit" value="Predict">
            </form>
        </div>
    </body>
    </html>
    """

# Function to preprocess uploaded image
def preprocess_image_file(file_stream):
    im = Image.open(file_stream).convert('L').resize((64, 64))
    arr = np.asarray(im, dtype=np.float32).reshape(1, -1)
    arr /= 255.0
    return arr

# Prediction endpoint with result display
@app.route('/predict', methods=['POST'])
def predict():
    f = request.files.get('file')
    if not f:
        return "No file uploaded", 400
    X = preprocess_image_file(f.stream)
    pred = clf.predict(X)[0]

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f2f2f2;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }}
            .card {{
                background: white;
                padding: 30px 40px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                text-align: center;
                width: 350px;
            }}
            .result {{
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }}
            a {{
                display: inline-block;
                margin-top: 15px;
                color: #007BFF;
                text-decoration: none;
                font-size: 14px;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Prediction Result</h2>
            <p class="result">Predicted Class: <b>{pred}</b></p>
            <a href="/">🔙 Go Back</a>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
