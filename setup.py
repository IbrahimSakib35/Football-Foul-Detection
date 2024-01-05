from pathlib import Path
import sys
file_path = Path(__file__).resolve()
root_path = file_path.parent
if root_path not in sys.path:
    sys.path.append(str(root_path))
ROOT = root_path.relative_to(Path.cwd())

ABOUT = 'About'
IMAGE = 'Image'
VIDEO = 'Video'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'
SOURCES_LIST = [ABOUT, IMAGE, VIDEO, RTSP, YOUTUBE]

IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'foul.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'foul_detected.jpg'

VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
VIDEO_4_PATH = VIDEO_DIR / 'video_4.mp4'
VIDEO_5_PATH = VIDEO_DIR / 'video_5.mp4'
VIDEO_6_PATH = VIDEO_DIR / 'video_6.mp4'
VIDEO_7_PATH = VIDEO_DIR / 'video_7.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
    'video_4': VIDEO_4_PATH,
    'video_5': VIDEO_5_PATH,
    'video_6': VIDEO_6_PATH,
    'video_7': VIDEO_7_PATH,
}

MODEL_DIR = ROOT / 'weights'
YOLOV5_MODEL = MODEL_DIR / 'yolov5.pt'
YOLOV8_MODEL = MODEL_DIR / 'yolov8.pt'