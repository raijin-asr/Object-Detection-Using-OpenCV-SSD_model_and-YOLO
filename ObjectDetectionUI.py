import ipywidgets as widgets
from IPython.display import display, clear_output
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

# Function to detect objects in an image
def detect_objects_in_image(change):
    clear_output()
    display(image_button, video_button, webcam_button)
    
    if change['new']:
        uploaded_file = change['owner']
        image = np.frombuffer(uploaded_file['content'], np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Perform object detection on the image
        ClassIndex, confidence, bbox = model.detect(image, confThreshold=0.55)
        
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(image, boxes, (255, 0, 0), 2)
                cv2.putText(image, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Function to detect objects in a video
def detect_objects_in_video(change):
    clear_output()
    display(image_button, video_button, webcam_button)
    
    if change['new']:
        uploaded_file = change['owner']
        video_path = io.BytesIO(uploaded_file['content'])
        
        # Perform object detection on the video
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
            
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if ClassInd <= 80:
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            
            cv2.imshow('Object Detection in Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Function to detect objects in webcam
def detect_objects_in_webcam(change):
    clear_output()
    display(image_button, video_button, webcam_button)
    
    if change['new']:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
            
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if ClassInd <= 80:
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            
            cv2.imshow('Object Detection in Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Load model and labels
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5) # 255/2=127.5
model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1,1]
model.setInputSwapRB(True) # auto convert BGR to RGB

classLabels = []
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# Create buttons for image, video, and webcam
image_button = widgets.FileUpload(accept='.jpeg, .jpg, .png', description='Image')
video_button = widgets.FileUpload(accept='.mp4', description='Video')
webcam_button = widgets.Button(description='Webcam')

# Attach event handlers to the buttons
image_button.observe(detect_objects_in_image, names='value')
video_button.observe(detect_objects_in_video, names='value')
webcam_button.on_click(detect_objects_in_webcam)

# Display the buttons
display(image_button, video_button, webcam_button)