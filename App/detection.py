'''
 Module for processing images and videos for object detection using YOLO model.
 Provides functions to handle image and video uploads, perform detection,
 and return annotated results along with relevant counts.
 '''

from Model.yolo import model
from App.utils import  save_video
from App.constants import *
import os
from PIL import Image
import cv2
from datetime import datetime



'''
 Function to process a single image
 Detect objects in an uploaded image and return:
    1. Annotated image with bounding boxes
    2. Counts of each detected object
'''
def process_image(uploaded_file, conf=CONFIDENCE_THRESHOLD):   
    
    image = Image.open(uploaded_file)   
    
    detection_results = model(image, conf=conf) # Run YOLO detection on the image    
   
    annotated_image = detection_results[0].plot()  
    
    counts = {} # Count how many objects of each class were detected
    for cls in detection_results[0].boxes.cls:
        class_name = model.names[int(cls)]  # Convert class index to name
        if class_name in counts:
            counts[class_name] += 1
        else:
            counts[class_name] = 1
    
    annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for PIL
    
    annotated_pil = Image.fromarray(annotated_rgb) # Convert numpy array to PIL Image
    
    original_name = os.path.splitext(uploaded_file.name)[0] # Get original file name without extension
    
    output_path = os.path.join(OUTPUT_IMAGE_FOLDER, f"{original_name}_annotated.jpg") # Define output path and give proper name
    
    annotated_pil.save(output_path) # Save annotated image
    
    return annotated_image, counts,detection_results[0].boxes # Return annotated image, counts, and boxes 


'''
Function to process a video
Detect objects in a video and return:
    1. Path to annotated video file
'''

def process_video(uploaded_file, conf=CONFIDENCE_THRESHOLD): 

    cap = cv2.VideoCapture(uploaded_file) # Open the uploaded video file    
    
    listof_annotated_frames = [] # List to store processed frames
    
    while cap.isOpened():
        frame_availability, frame = cap.read()
        if not frame_availability:
            break  # End of video
              
        results = model(frame, conf=conf, stream=True)  # Run YOLO detection on the current frame       
    
        annotated_frame = frame # Start with the original frame
        
        # Annotate detected objects and check for violations
        for r in results:
            annotated_frame = r.plot()  # Draw boxes and labels
                  
       
        listof_annotated_frames.append(annotated_frame)    
  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"detected_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_VIDEO_FOLDER, output_filename)

    video_output_path = save_video(listof_annotated_frames, output_path)   # Save all frames as a new video
        
    cap.release()# Release the video capture object   
    
    return video_output_path  # Return video path 
