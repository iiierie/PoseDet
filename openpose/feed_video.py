import cv2
import numpy as np
from openpose.detect import *

# Load the body pose estimation model
model = Body('body_pose_model.pth')

# Load the video file
video_path = 'cut_jog.mp4'
cap = cv2.VideoCapture(video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Input video FPS:", fps)
print("Input video width:", width)
print("Input video height:", height)

# Create a video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

print("Output video FPS:", fps)
print("Output video width:", width)
print("Output video height:", height)

print("Processing video...")

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
batch_size = 32

while True:
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if not frames:
        break

    # Preprocess the frames
    preprocessed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (368, 368))
        frame = frame / 256.0 - 0.5
        preprocessed_frames.append(frame)

    # Estimate body pose for each frame
    candidates = []
    subsets = []
    for frame in preprocessed_frames:
        candidate, subset = model(frame)
        candidates.append(candidate)
        subsets.append(subset)

    # Draw the body pose on each frame
    canvases = []
    for i, frame in enumerate(preprocessed_frames):
        canvas = draw_bodypose(frame, candidates[i], subsets[i])
        canvas = cv2.resize(canvas, (width, height))
        canvases.append(canvas)

    # Write the output frames to the video writer
    for canvas in canvases:
        out.write(canvas)
        print("Writing frame to output video")

    frame_count += len(frames)
    print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.2f}%)", end='\r')

cap.release()
out.release()
cv2.destroyAllWindows()
print("\nVideo processing complete!")