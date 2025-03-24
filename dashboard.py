import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
import subprocess
from enum import Enum
from pathlib import Path

# Add the directory containing dashboard.py (and helper.py) to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.tracker import do_tracking_evaluation
from argparse import Namespace

def reencode_video(input_path, output_path):
    """
    Reencode a video file using FFmpeg to H.264 (libx264) and AAC.
    Adjust presets and CRF (quality) as needed.
    """
    cmd = [
        "ffmpeg",
        "-y",                 # Overwrite output file if it exists
        "-i", input_path,     # Input file
        "-c:v", "libx264",    # Use H.264 video codec
        "-preset", "fast",    # Encoding speed/quality trade-off
        "-crf", "23",         # Constant Rate Factor (lower means better quality)
        "-c:a", "aac",        # Use AAC for audio (if needed)
        output_path
    ]
    subprocess.run(cmd, check=True)

def evaluate(**kwargs):
    if not os.path.exists(kwargs["checkpoint_path"]):
        raise Exception(f"Error: Model path '{kwargs['checkpoint_path']}' does not exist")

    metrics = do_tracking_evaluation(Namespace(**kwargs))    
    return metrics 

st.title("Computer Vision Demo - Vehicle Tracker Dashboard")

class ModelOptions(Enum):
    SEGFORMER_IMAGE_DIFF = "SegFormer (Image Diff)"
    SEGFORMER_OPT_FLOW = "SegFormer (Optical Flow)"
    SEGFORMER_IMAGE_SEQ = "SegFormer (Image Sequence)"

# Sidebar for setting training parameters
st.sidebar.header("Evaluation Parameters")
model = st.sidebar.selectbox("Detector Model", options=[ModelOptions.SEGFORMER_IMAGE_DIFF.value, ModelOptions.SEGFORMER_OPT_FLOW.value, ModelOptions.SEGFORMER_IMAGE_SEQ.value])
number_of_vehicles = st.sidebar.number_input("Number of vehicles", min_value=1, max_value=10, value=1, step=1)
smooth_tracking = st.sidebar.checkbox("Enable Smoothed Tracking", value=True)
detection_tracking = st.sidebar.checkbox("Enable Detection Tracking", value=False)
segmentation = st.sidebar.checkbox("Enable Segmentation", value=False)

st.write("Adjust the training parameters in the sidebar and then click the **Start Training** button.")

if st.button("Evaluate"):
    st.write("Starting evaluation...")
    # Initialize placeholders for progress and status
    st.session_state.progress_bar = st.progress(0)
    st.session_state.status_text = st.empty()

    model_path = None
    if model == ModelOptions.SEGFORMER_IMAGE_DIFF.value:
        model_path = "./checkpoints/model_diff_grayscale_0_935.ckpt"
    elif model == ModelOptions.SEGFORMER_OPT_FLOW.value:
        model_path = "./checkpoints/model_optical_flow_0_94.ckpt"
    elif model == ModelOptions.SEGFORMER_IMAGE_SEQ.value:
        model_path = "./checkpoints/model_3frame_iou_0_93.ckpt"

    metrics = evaluate(**{
        "checkpoint_path": model_path,
        "load": f"./{Path(str(model)).name.replace('.ckpt', '')}_regions.pth",
        "enable_smooth_tracking": smooth_tracking,
        "enable_detection_tracking": detection_tracking,
        "enable_segmentation": segmentation,
        "num_vehicles": number_of_vehicles,
        "progress_callback": lambda mesg, perc: (st.session_state.progress_bar.progress(perc), st.session_state.status_text.text(mesg))
    })

    st.subheader("IoU Score")
    st.write(f"Generating video...")
    
    output_path = "./reencoded_video.mp4"
    reencode_video(metrics["output_video_path"], output_path)
    st.video(output_path)
    #st.video("./reencoded_video.mp4")

    st.subheader("IoU Score")
    st.write(f"The Intersection over Union (IoU) score is: **{metrics['mean_detection_iou']:.4f}**")
    
    
    