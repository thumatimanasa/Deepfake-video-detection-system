import cv2
import h5py
import numpy as np

# Open video file
video_path = "C:/Users/venkat surisetti/Desktop/face_swap_detector/deepfake_model.h5.mp4"
cap = cv2.VideoCapture(video_path)

# Create HDF5 file
h5_f ile = h5py.File("C:/Users/venkat surisetti/Desktop/face_swap_detector/deepfake_model_new.h5", "w")

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)  # Store frame as a NumPy array

cap.release()

# Convert frames to NumPy array
frames = np.array(frames)

# Save to .h5 file
h5_file.create_dataset("video_frames", data=frames)
h5_file.close()

print("MP4 successfully converted to HDF5 format!")
