"""
Object Detection Streamlit Web App
====================================
A beginner-friendly web interface for detecting objects in images and videos
using YOLOv8.

Author: Rohit Jha
Date: 26 Oct 2025
"""


import streamlit as st
import cv2
import tempfile
import os 
from datetime import datetime

# Import your custom modules
from App.detection import *
from App.utils import *
from App.constants import *

# ========================
# PAGE CONFIG
# ========================

st.set_page_config(
    page_title="Object Detection System",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========================
# CUSTOM CSS
# ========================

st.markdown("""
<style>
.main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
.sub-header {font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem;}
.stButton>button {width: 100%; background-color: #1f77b4; color: white; border-radius: 5px; padding: 0.5rem; font-weight: bold;}
.stButton>button:hover {background-color: #155a8a;}
</style>
""", unsafe_allow_html=True)

# ========================
# HEADER
# ========================

st.markdown('<div class="main-header">🔎 Object Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detect any objects in real-time</div>', unsafe_allow_html=True)


# ========================
# SIDEBAR
# ========================

st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

input_type = st.sidebar.radio("📂 Select Input Type:", ["Image", "Video"])
confidence = st.sidebar.slider("🎯 Confidence Threshold:", 0.0, 1.0, float(CONFIDENCE_THRESHOLD), 0.05)
st.sidebar.markdown("---")

with st.sidebar.expander("📖 Instructions", expanded=False):
    st.markdown("""
    1. Select Input Type: Image or Video
    2. Adjust Confidence Threshold
    3. Upload your file
    4. View detection results
    5. Results are automatically saved
    """)

with st.sidebar.expander("ℹ️ About", expanded=False):
    st.markdown("""
    **Detection System v1.0**  
    - Real-time object detection  
    - Confidence threshold adjustment  
    - Automatic result saving
    """)

st.sidebar.markdown("*Made with ❤️ by Rohit Jha*")

# ========================
# CREATE OUTPUT FOLDERS
# ========================

os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_VIDEO_FOLDER, exist_ok=True)


# ========================
# IMAGE PROCESSING
# ========================

if input_type == "Image":
    st.subheader("📷 Image Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            st.info("🔄 Processing image... Please wait.")
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", use_container_width=True)

            with st.spinner("🔍 Detecting Objects..."):
                annotated_image, detections,boxes  = process_image(uploaded_file, confidence)

            with col2:
                st.image(annotated_image, caption="Detected Image", use_container_width=True)

            st.success("✅ Image processed successfully!")

            # ========================
            # DISPLAY DETECTIONS
            # ========================
            if detections:
                st.subheader("📊 Detection Results")
                cols = st.columns(min(len(detections), 4))
                for idx, (cls, count) in enumerate(detections.items()):
                    with cols[idx % len(cols)]:
                        st.metric(label=cls, value=count)

                # Optional: detailed table
                st.markdown("### 📋 Detection Counts Table")
                table_data = [{"Class": cls, "Count": cnt} for cls, cnt in detections.items()]
                st.table(table_data)


            else:
                st.info("ℹ️ No objects detected. Try lowering the confidence threshold.")

            # Save annotated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_IMAGE_FOLDER, f"detected_{timestamp}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            st.success(f"💾 Image saved to: `{output_path}`")

        except Exception as e:
            st.error("❌ An error occurred while processing the image.")
            st.error(f"Details: {str(e)}")
            import traceback
            traceback.print_exc()

    else:
        st.info("👆 Please upload an image to get started")



# ========================
# VIDEO PROCESSING
# ========================


elif input_type == "Video":
    st.subheader("🎥 Video Detection")

    uploaded_video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video_file is not None:
        # Display uploaded video
        st.markdown("### 📥 Original Video")
        st.video(uploaded_video_file)

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_video_file.read())
            temp_video_path = temp_video.name

        st.info("🔄 Processing video... This may take a few minutes depending on length.")

        try:
            with st.spinner("🔍 Running object detection on video..."):
                output_video_path= process_video(temp_video_path, conf=confidence)

            st.success("✅ Video processed successfully!")

            # Display results
            st.markdown("### 📤 Processed Video")
            try:
                st.video(output_video_path)
            except Exception as e:
                st.error("❌ Cannot display video inline. You can download it instead.") 
                          
            st.info("ℹ️ Note: The model processes videos at 24 FPS. Videos with higher FPS may appear longer after processing.")
            col1,col2= st.columns(2)         

            # Compute video duration
            cap = cv2.VideoCapture(output_video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps else 0
            cap.release()
            col1.metric("🎬 FPS", f"{fps}")
            col2.metric("⏱️ Video Duration", f"{duration}s")

            # Allow user to download processed video
            st.download_button(
                label="📥 Download Processed Video",
                data=open(output_video_path, "rb").read(),
                file_name=os.path.basename(output_video_path),
                mime="video/mp4"
            )

        except Exception as e:
            st.error("❌ An error occurred while processing the video.")
            st.error(f"Details: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

    else:
        st.info("👆 Please upload a video to get started")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888; padding:2rem;'>Object Detection System | Powered by YOLOv8 and Streamlit</div>", unsafe_allow_html=True)
