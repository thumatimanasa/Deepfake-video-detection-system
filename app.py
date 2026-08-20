# app.py
from flask import Flask, request, render_template
import os
from model import DeepFakeDetector
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load your pre-trained model
detector = DeepFakeDetector('deepfake_model_new1.h5')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and file.filename.endswith('.mp4'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Call the model to predict
        result = detector.predict(file_path)
        return f'The video is: {result}'
    else:
        return "Invalid file format. Please upload an .mp4 file."

if __name__ == '__main__':
    app.run(debug=True)