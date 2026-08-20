import cv2
import numpy as np
import tensorflow as tf

class DeepFakeDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fake_frames = 0
        real_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))  # Resize to model input size
            frame = frame / 255.0  # Normalize pixel values
            frame = np.expand_dims(frame, axis=0)  # Add batch dimension

            prediction = self.model.predict(frame)[0][0]  # Get the prediction score

            if prediction > 0.5:
                fake_frames += 1
            else:
                real_frames += 1

            frame_count += 1
        cap.release()

        if frame_count == 0:
            return "Error: No frames found in video."

        # Determine the final result based on majority voting
        return "real" if fake_frames > real_frames else "fake"
