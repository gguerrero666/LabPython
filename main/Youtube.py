from ultralytics import YOLO

ytQuality = "?vq=small"
yt = "https://youtu.be/8DpjANtV-bs" + ytQuality

model = YOLO("yolo/yolov8n.pt")
results = model.track(source=yt, conf=0.3, iou=0.5, show=True, verbose=False)
