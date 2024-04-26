import cv2
from ultralytics import YOLO
import subprocess

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

result= model("lambo2.mp4", show= True, save=False)


# Check for key events
while True:
    key = cv2.waitKey(1) & 0xFF

    # If 'q' key is pressed, break the loop and run the Ctrl+C command in the terminal
    if key == ord('q'):
        # result.show()  # Close the video window
        subprocess.run(["kill", "-SIGINT", str(subprocess.Popen(["pidof", "python"], stdout=subprocess.PIPE).communicate()[0].strip())])
        break



"""
# Open the video file
video_path = "lambo.mp4"
capt = cv2.VideoCapture(video_path)

# Check if the video opened correctly
if not capt.isOpened():
    raise IOError("Cannot open video")

# Set the title name of the frame window
title = "Object Detection in Webcam using YOLO"

while True:
    # Read a frame from the video
    ret, frame = capt.read()
    if not ret:
        break

    # Perform object detection on the frame
    detections = model(frame)

    # Render each detection individually
    for detection in detections.pred:
        image = detection.render()

        # Display the frame with detections
        cv2.imshow(title, image)

    # Check for key events
    key = cv2.waitKey(1)
    
    # Break the loop if 'q' is pressed
    if key & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
capt.release()
cv2.destroyAllWindows()
"""