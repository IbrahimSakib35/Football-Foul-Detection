from pathlib import Path
import PIL
import streamlit as st
import setup
import mode
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Football Foul Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Football Foul Detection")
st.sidebar.header("Model Configurations")
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100
model_type = st.sidebar.radio(
    "Select Model", ['YOLOv5', 'YOLOv8'])
if model_type == 'YOLOv5':
    model_path = Path(setup.YOLOV5_MODEL)
elif model_type == 'YOLOv8':
    model_path = Path(setup.YOLOV8_MODEL)

try:
    model = mode.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model.Check the specified path: {model_path}")
    st.error(ex)
# st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", setup.SOURCES_LIST)

def dehaze_with_dcp(image):

    image_np = np.array(image)
    image_np = image_np.astype(np.float64) / 255.0
    dark_channel = np.min(image_np, axis=-1)
    atmospheric_light = np.percentile(dark_channel, 80)
    transmission = 1 - 0.70 * dark_channel / atmospheric_light
    transmission = np.expand_dims(transmission, axis=-1)
    dehazed_image_np = (image_np - atmospheric_light) / np.maximum(transmission, 0.1) + atmospheric_light
    dehazed_image_np = np.clip(dehazed_image_np, 0, 1)
    dehazed_image = Image.fromarray((dehazed_image_np * 255).astype(np.uint8))

    return dehazed_image

source_img = None
#selecting image
if source_radio == setup.ABOUT:
    st.write(
            """
            Hello 👋, We are **:red[Group: T2310081]**
            """
            """
            Our Thesis Topic: **:green[Real-Time Foul Detection in Football Matches using Machine Learning Techniques]**
            """
            """
            Select From Left Side Menus to detect **:blue[Foul, Fouling_Player and Victim]** from **:violet[Images, Videos, Real Time Streaming Protocol(RTSP) and Youtube Videos]**
            """
        )
elif source_radio == setup.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(setup.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(setup.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            apply_dehazing = st.sidebar.checkbox('Apply Dehazing', value=True)

            if st.sidebar.button('Detect Foul'):
                # Apply dehazing if enabled
                dehazed_image = dehaze_with_dcp(uploaded_image) if apply_dehazing else uploaded_image

                # Perform YOLOv8 inference on the (dehazed) image
                res = model.predict(dehazed_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                        use_column_width=True)

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == setup.VIDEO:
    mode.play_stored_video(confidence, model)
elif source_radio == setup.RTSP:
    mode.play_rtsp_stream(confidence, model)
elif source_radio == setup.YOUTUBE:
    mode.play_youtube_video(confidence, model)
else:
    st.error("Please select a valid source type!")