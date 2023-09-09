from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("../yolo/yolov8n.pt")
results = model.track(source="https://youtu.be/8DpjANtV-bs", conf=0.3, iou=0.5, show=True,
                      imgsz=640, verbose=True)
