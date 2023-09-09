from ultralytics import YOLO
import cv2

# load yolov8 model
model = YOLO('yolo/yolov8m.pt')

# Open the video file
video_path = '../sample/p3.mp4'
cap = cv2.VideoCapture(video_path)
success = True

# Loop through the video frames
while success:
    # Read a frame from the video
    success, frame = cap.read()
    if success:

        # detect objects & track objects
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=0, verbose=False)

        # print ids
        for result in results:
            print(result.boxes.id)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # visualize
        cv2.imshow('frame', annotated_frame)

        # Terminate run when "ESC" pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
