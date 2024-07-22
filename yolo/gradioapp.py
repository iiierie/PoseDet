import cv2
import gradio as gr
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load the YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

def process_frame(frame):
    # Run pose estimation
    results = model(frame)
    
    # Get the plot image with poses drawn on it
    plot = results[0].plot()
    return plot

def process_image(image):
    # Convert PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed_image = process_frame(frame)
    # Convert back to PIL image for Gradio output
    return Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

def process_video(video):
    # Process video and save the result
    cap = cv2.VideoCapture(video.name)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        frames.append(processed_frame)
    
    cap.release()
    
    # Convert frames to video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "output_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return output_path

def process_webcam_frame(frame):
    # Convert PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    processed_frame = process_frame(frame)
    return Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

def webcam_loop():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        # Convert to PIL Image
        processed_image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        yield processed_image

    cap.release()

with gr.Blocks() as demo:
    gr.Markdown("# YOLOv8 Pose Estimation")
    
    with gr.Tab("Upload Image"):
        image_input = gr.Image(type="pil")
        image_output = gr.Image(type="pil")
        image_input.change(process_image, inputs=image_input, outputs=image_output)
    
    with gr.Tab("Upload Video"):
        video_input = gr.File()
        video_output = gr.File()
        video_input.change(process_video, inputs=video_input, outputs=video_output)
    
    with gr.Tab("Webcam"):
        # Use a streaming component for webcam
        webcam_output = gr.Image(type="pil")
        gr.Interface(fn=lambda: next(webcam_loop()), 
                     inputs=[], 
                     outputs=webcam_output, 
                     live=True).launch(share=True)

demo.launch()
