import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("model/best.pt")
names = model.model.names

# Open video capture
cap = cv2.VideoCapture("sample_videos/sheep1.mp4")
if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create video writer
out = cv2.VideoWriter("segmentation-outcome/instance-segmentation.mp4", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
if not out.isOpened():
    print("Error: Couldn't create video file.")
    cap.release()
    exit()

# Main loop for video processing
while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform prediction
    results = model.predict(im0)
    annotated_frame = results[0].plot()

    # Write annotated frame to output video
    out.write(annotated_frame)

    # Display annotated frame
    cv2.imshow("instance-segmentation", annotated_frame)

    # Check for key press to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
