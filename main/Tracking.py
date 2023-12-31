from ultralytics import YOLO
import cv2

# load yolov8 model
model = YOLO('yolo/yolov8n.pt')

# load video
video_path = '../sample/p3.mp4'
cap = cv2.VideoCapture(video_path)
ret = True

# read frames
while ret:
    ret, frame = cap.read()
    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True, classes=0, verbose=False)

        # plot results
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
