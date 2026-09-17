import streamlit as st
import cv2
import numpy as np
import os
import sys
import tempfile
import time

# ---------------------------------------------------------
# PROJECT MODULE IMPORT
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipeline import RoadSafetyPipeline


# ---------------------------------------------------------
# STREAMLIT PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Road Vision Safety System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# PIPELINE INITIALIZATION
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading Computer Vision Model...")
def initialize_system():
    return RoadSafetyPipeline()


cv_system = initialize_system()


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def convert_to_rgb(frame):
    """Convert OpenCV BGR image into RGB format for Streamlit."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def show_metric_panel(output):
    """Display the main analysis measurements."""

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            label="🚗 Vehicles",
            value=output["vehicle_count"]
        )

    with b:
        st.metric(
            label="🎯 Avg. Confidence",
            value=f"{output['avg_conf']:.2f}"
        )

    with c:
        st.metric(
            label="〰️ Lane Segments",
            value=output["lines_detected"]
        )

    with d:
        st.metric(
            label="⏱️ Processing",
            value=f"{output['process_time']:.3f} sec"
        )


def display_processing_steps(original, data):
    """Render all intermediate computer vision outputs."""

    stages = [
        (
            "1. Input Frame",
            original,
            "Original road image provided to the system."
        ),
        (
            "2. Grayscale Representation",
            data["gray"],
            "The image is converted into grayscale for intensity-based processing."
        ),
        (
            "3. Noise Reduction",
            data["blur"],
            "Gaussian filtering smooths the image and reduces unwanted noise."
        ),
        (
            "4. Edge Map",
            data["edges"],
            "Canny edge detection identifies important image boundaries."
        ),
        (
            "5. Road ROI",
            data["roi_mask"],
            "A Region of Interest is selected to concentrate on the road area."
        ),
        (
            "6. ROI Edge Map",
            data["cropped_edges"],
            "Only useful edge information inside the selected road region is retained."
        )
    ]

    for position in range(0, len(stages), 2):

        left, right = st.columns(2)

        title, picture, description = stages[position]

        with left:
            st.image(
                convert_to_rgb(picture)
                if len(picture.shape) == 3
                else picture,
                use_container_width=True
            )
            st.markdown(f"**{title}**")
            st.caption(description)

        if position + 1 < len(stages):

            title, picture, description = stages[position + 1]

            with right:
                st.image(
                    convert_to_rgb(picture)
                    if len(picture.shape) == 3
                    else picture,
                    use_container_width=True
                )
                st.markdown(f"**{title}**")
                st.caption(description)


# ---------------------------------------------------------
# APPLICATION HEADER
# ---------------------------------------------------------

st.title("🚦 AI-Based Road Safety Vision System")

st.write(
    "A Computer Vision based academic system for analyzing road scenes, "
    "detecting vehicles, identifying lane structures and performing "
    "basic spatial road-safety analysis."
)

st.info(
    "🎓 **Academic Project — B.Tech Computer Science (AI & ML)**\n\n"
    "The system demonstrates how image processing and object detection "
    "techniques can be combined for road-scene understanding."
)


# ---------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------

st.sidebar.title("⚙️ System Controls")

source_mode = st.sidebar.selectbox(
    "Select Input",
    ["Road Image", "Road Video"]
)

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.40,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Adjust the confidence value to control the minimum confidence "
    "required for vehicle detections."
)


# =========================================================
# IMAGE ANALYSIS MODE
# =========================================================

if source_mode == "Road Image":

    st.subheader("📷 Road Image Analysis")

    image_file = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing a road scene."
    )

    if image_file:

        # Read uploaded file
        raw_data = image_file.getvalue()

        image_array = np.frombuffer(
            raw_data,
            dtype=np.uint8
        )

        original_frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if original_frame is None:

            st.error(
                "The uploaded file could not be interpreted as an image."
            )

        else:

            # ---------------------------------------------
            # RUN COMPUTER VISION PIPELINE
            # ---------------------------------------------

            with st.spinner("Running road-scene analysis..."):

                analysis = cv_system.process_frame(
                    original_frame,
                    confidence
                )

            # ---------------------------------------------
            # RESULTS
            # ---------------------------------------------

            st.divider()

            st.subheader("📊 Analysis Results")

            show_metric_panel(analysis)

            # ---------------------------------------------
            # PROCESSING PIPELINE
            # ---------------------------------------------

            st.divider()

            st.subheader(
                "🔬 Computer Vision Processing Stages"
            )

            display_processing_steps(
                original_frame,
                analysis["intermediates"]
            )

            # ---------------------------------------------
            # FINAL OUTPUT
            # ---------------------------------------------

            st.divider()

            st.subheader("🚘 Final Road Safety Analysis")

            final_output = convert_to_rgb(
                analysis["final_image"]
            )

            st.image(
                final_output,
                use_container_width=True
            )

            st.caption(
                "Final output containing detected vehicles, lane "
                "information, ROI boundaries and spatial references."
            )


# =========================================================
# VIDEO ANALYSIS MODE
# =========================================================

else:

    st.subheader("🎥 Road Video Analysis")

    video_file = st.file_uploader(
        "Choose a road video",
        type=["mp4", "avi"],
        help="Upload a road video for frame-by-frame analysis."
    )

    if video_file:

        # ---------------------------------------------
        # TEMPORARY VIDEO STORAGE
        # ---------------------------------------------

        temporary_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temporary_file.write(
            video_file.getvalue()
        )

        temporary_file.close()

        video_path = temporary_file.name

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():

            st.error("Unable to open the uploaded video.")

        else:

            # ---------------------------------------------
            # VIDEO INFORMATION
            # ---------------------------------------------

            total_frames = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            video_fps = capture.get(
                cv2.CAP_PROP_FPS
            )

            st.write(
                f"**Video Frames:** {total_frames}  |  "
                f"**Source FPS:** {video_fps:.1f}"
            )

            # ---------------------------------------------
            # LIVE RESULT AREA
            # ---------------------------------------------

            video_display = st.empty()

            st.divider()

            st.subheader("📈 Live Detection Measurements")

            fps_box, vehicle_box, confidence_box, lane_box = st.columns(4)

            fps_display = fps_box.empty()
            vehicle_display = vehicle_box.empty()
            confidence_display = confidence_box.empty()
            lane_display = lane_box.empty()

            stop_processing = st.sidebar.button(
                "⛔ Stop Analysis"
            )

            # ---------------------------------------------
            # PROCESS VIDEO FRAMES
            # ---------------------------------------------

            frame_number = 0

            while capture.isOpened():

                if stop_processing:
                    break

                success, current_frame = capture.read()

                if not success:
                    break

                frame_number += 1

                result = cv_system.process_frame(
                    current_frame,
                    confidence
                )

                # Display processed frame
                video_display.image(
                    convert_to_rgb(
                        result["final_image"]
                    ),
                    use_container_width=True
                )

                # Calculate approximate processing FPS
                elapsed = result["process_time"]

                current_fps = (
                    1.0 / elapsed
                    if elapsed > 0
                    else 0
                )

                # Update live measurements
                fps_display.metric(
                    "⚡ Processing FPS",
                    f"{current_fps:.1f}"
                )

                vehicle_display.metric(
                    "🚗 Vehicles",
                    result["vehicle_count"]
                )

                confidence_display.metric(
                    "🎯 Confidence",
                    f"{result['avg_conf']:.2f}"
                )

                lane_display.metric(
                    "〰️ Lane Segments",
                    result["lines_detected"]
                )

            capture.release()

            # Remove temporary video
            try:
                os.remove(video_path)
            except OSError:
                pass

            st.success(
                f"Video analysis completed. "
                f"{frame_number} frames processed."
            )


# =========================================================
# PROJECT LIMITATIONS
# =========================================================

st.divider()

with st.expander("ℹ️ System Limitations & Academic Disclaimer"):

    limitation_points = [
        "Lane markings that are faded, damaged or blocked may reduce lane-detection performance.",
        "Lighting, shadows, rain and other weather conditions can affect image processing.",
        "Vehicle detection accuracy depends on the trained object-detection model.",
        "Road-safety indications are generated using predefined visual-analysis rules.",
        "The system is intended for academic demonstration and is not a certified ADAS or autonomous-driving system."
    ]

    for item in limitation_points:
        st.markdown(f"• {item}")