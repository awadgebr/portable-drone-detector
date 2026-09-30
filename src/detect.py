from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("test_images/test.jpg", save=True)

print("Detection complete!")