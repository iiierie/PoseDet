import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import io
import streamlit.components.v1 as components
# from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Load YOLO models for different tasks
pose_model = YOLO("yolov8n-pose.pt")
seg_model = YOLO("yolov8n-seg.pt")
det_model = YOLO("yolov8n.pt")

def detect_pose(image):
    image = np.array(image)
    results = pose_model(image)
    plot = results[0].plot()
    return Image.fromarray(plot)

def detect_objects(image):
    image = np.array(image)
    results = det_model(image)
    plot = results[0].plot()
    return Image.fromarray(plot)

def segment_objects(image):
    image = np.array(image)
    results = seg_model(image)
    plot = results[0].plot()
    return Image.fromarray(plot)

def process_video(video_bytes, task):
    temp_input_file_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_input_file_path.write(video_bytes)
    temp_input_file_path.seek(0)
    
    temp_output_file_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    
    try:
        cap = cv2.VideoCapture(temp_input_file_path.name)
        if not cap.isOpened():
            raise ValueError("Error opening video file for input")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_output_file_path.name, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError("Error opening video file for output")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if task == "Pose Estimation":
                results = pose_model(frame_rgb)
            elif task == "Object Detection":
                results = det_model(frame_rgb)
            elif task == "Segmentation":
                results = seg_model(frame_rgb)
            
            plot = results[0].plot()
            plot_bgr = cv2.cvtColor(np.array(plot), cv2.COLOR_RGB2BGR)
            out.write(plot_bgr)
        
        cap.release()
        out.release()
    except Exception as e:
        print(f"Error during video processing: {e}")
        return None

    return temp_output_file_path.name

class VideoProcessor(VideoProcessorBase):
    def __init__(self, task):
        self.task = task

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.task == "Pose Estimation":
            results = pose_model(img_rgb)
        elif self.task == "Object Detection":
            results = det_model(img_rgb)
        elif self.task == "Segmentation":
            results = seg_model(img_rgb)
        
        plot = results[0].plot()
        plot_bgr = cv2.cvtColor(np.array(plot), cv2.COLOR_RGB2BGR)
        
        return plot_bgr


def main():
    st.title("Detection and Segmentation App")
    
    if 'output_video_path' not in st.session_state:
        st.session_state.output_video_path = None
    
    task = st.sidebar.selectbox("Choose Task", ["Pose Estimation", "Object Detection", "Segmentation"])
    
    option = st.sidebar.selectbox("Choose Input Source", ["Upload Image", "Upload Video", "Webcam"])
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("")
            st.write(f"Running {task.lower()}...")

            result_image = None  # Initialize result_image

            if task == "Pose Estimation":
                result_image = detect_pose(image)
            elif task == "Object Detection":
                result_image = detect_objects(image)
            elif task == "Segmentation":
                result_image = segment_objects(image)
            
            if result_image is not None:  # Check if result_image is not None
                st.image(result_image, caption=f"{task} Result", use_column_width=True)
                buffered = io.BytesIO()
                result_image.save(buffered, format="PNG")
                st.download_button(label="Download Result Image", data=buffered.getvalue(), file_name=f"{task.lower().replace(' ', '_')}_result.png")
            else:
                st.write("Error: Unable to process image.")
    
    elif option == "Upload Video":
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4"])
        if uploaded_video is not None:
            if st.session_state.output_video_path is None:
                st.write(f"Processing video for {task.lower()}...")
                st.session_state.output_video_path = process_video(uploaded_video.getvalue(), task)
                
            if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
                st.write("Video processing complete.")
                with open(st.session_state.output_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                # st.video(video_bytes, format="video/mp4")
                st.download_button(label="Download Processed Video", data=video_bytes, file_name=f"{task.lower().replace(' ', '_')}_result.mp4")
                
                if os.path.exists(st.session_state.output_video_path):
                    os.remove(st.session_state.output_video_path)


    # normal opencv
    # elif option == "Webcam":
    #     st.write("Opening webcam...")
    #     stframe = st.empty()
    #     stop_button = st.button('Stop Webcam', key='stop_webcam')
        
    #     cap = cv2.VideoCapture(0)
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             st.write("Failed to grab frame")
    #             break
            
    #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
    #         if task == "Pose Estimation":
    #             results = pose_model(frame_rgb)
    #         elif task == "Object Detection":
    #             results = det_model(frame_rgb)
    #         elif task == "Segmentation":
    #             results = seg_model(frame_rgb)
            
    #         plot = results[0].plot()
    #         stframe.image(plot, channels="RGB")
            
    #         if stop_button:
    #             break
        
    #     cap.release()

    # webrtc
    elif option == "Webcam":
            st.write("Opening webcam...")
            webrtc_streamer(key="example", video_processor_factory=lambda: VideoProcessor(task))


if __name__ == "__main__":
    main()





