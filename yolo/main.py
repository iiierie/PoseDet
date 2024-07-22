import cv2
from ultralytics import YOLO

# Load the YOLOv8 pose model
model = YOLO("yolov8s-cls.pt")

# Initialize the webcam (default camera index is 0)
# cap = cv2.VideoCapture("jog.mp4")
# cap = cv2.VideoCapture("../moments_in_time/lifting/7-4-8-9-0-1-8-6-25674890186_4.mp4")
# cap = cv2.VideoCapture("moments_in_time/lifting/7-1-8-8-7-7-3-2-21671887732_2.mp4")
cap = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run pose estimation
    results = model(frame)

    # Get the plot image with poses drawn on it
    plot = results[0].plot()

    # Display the resulting frame
    cv2.imshow('Pose Estimation', plot)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
