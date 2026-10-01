import cv2
from pathlib import Path
from mcos_decoder import load_groundtruth

VIDEO_PATH = Path("raw_dataset/Data/Video_V/V_DRONE_001.mp4")
LABEL_PATH = Path("raw_dataset/Data/Video_V/V_DRONE_001_LABELS.mat")

IMAGE_DIR = Path("dataset/images/train")
LABEL_DIR = Path("dataset/labels/train")

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Drone is class 0 in data.yaml
CLASS_ID = 0

bboxes = load_groundtruth(str(LABEL_PATH))
video = cv2.VideoCapture(str(VIDEO_PATH))

frame_number = 0
saved = 0

while True:
    success, frame = video.read()

    if not success:
        break

    # We confirmed that video frame 0 matches bboxes[1]
    bbox_index = frame_number + 1

    if frame_number % 30 == 0 and bbox_index < len(bboxes):
        bbox = bboxes[bbox_index]

        if bbox is not None:
            x, y, w, h = bbox

            image_height, image_width = frame.shape[:2]

            # Convert (x, y, width, height) to YOLO format:
            # center_x, center_y, width, height — all normalized 0 to 1
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            norm_w = w / image_width
            norm_h = h / image_height

            name = f"V_DRONE_001_{frame_number:06d}"

            image_path = IMAGE_DIR / f"{name}.jpg"
            label_path = LABEL_DIR / f"{name}.txt"

            cv2.imwrite(str(image_path), frame)

            with open(label_path, "w") as f:
                f.write(
                    f"{CLASS_ID} "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{norm_w:.6f} "
                    f"{norm_h:.6f}\n"
                )

            saved += 1

    frame_number += 1

video.release()

print(f"Processed {frame_number} video frames.")
print(f"Created {saved} YOLO training images.")
print(f"Created {saved} YOLO label files.")