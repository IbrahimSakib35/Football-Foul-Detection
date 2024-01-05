from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import setup
from app import dehaze_with_dcp

def load_model(model_path):
    model = YOLO(model_path)
    return model

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False

    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def dehaze_option():
    dehaze = st.radio("Dehazing", ('Yes', 'No'))
    apply_dehazing = True if dehaze == 'Yes' else False
    return apply_dehazing

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None, dehazer=None):
    image = cv2.resize(image, (720, int(720*(9/16))))

    if dehazer:
        image = dehaze_with_dcp(image)

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    is_display_tracker, tracker = display_tracker_options()
    apply_dehazing = dehaze_option()

    if st.sidebar.button('Detect Foul'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             apply_dehazing
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_rtsp_stream(conf, model):
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.0.100:400/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    apply_dehazing = dehaze_option()

    if st.sidebar.button('Detect Foul'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             apply_dehazing
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video", setup.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()
    apply_dehazing = dehaze_option()

    with open(setup.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()

    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Foul'):
        try:
            vid_cap = cv2.VideoCapture(str(setup.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             apply_dehazing
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
