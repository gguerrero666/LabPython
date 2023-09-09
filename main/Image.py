from ultralytics import YOLO
import cv2

model = YOLO('../yolo/yolov8x.pt')
# results = model("images/1.jpg", show=True)
results = model("images/test-img.png", show=True)
cv2.waitKey(0)
