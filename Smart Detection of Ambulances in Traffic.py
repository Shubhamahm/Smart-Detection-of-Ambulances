import torch
import cv2
import numpy as np
import time
from pathlib import Path

def load_model(weights='yolov5s.pt'):
    """Load YOLOv5 model."""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
    return model

def detect_ambulance(model, frame):
    """Detect ambulance and its key features in the given frame."""
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get detection results
    
    ambulance_detected = False
    
    for _, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Define ambulance-related objects
        ambulance_features = ['ambulance', 'cross sign', 'siren', 'horn', 'text']
        
        if label in ambulance_features and confidence > 0.5:
            ambulance_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, ambulance_detected

def process_video(video_path, model):
    """Process video for ambulance detection."""
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, detected = detect_ambulance(model, frame)
        
        if detected:
            print("Ambulance detected!")
        
        cv2.imshow('Ambulance Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    yolov5_model_path = 'path/to/custom_yolov5_weights.pt' 
    model = load_model(yolov5_model_path)
    video_file = 'path/to/traffic_video.mp4'  
    process_video(video_file, model)
