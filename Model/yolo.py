'''
THIS FILE LOADS THE YOLO MODEL ONCE AND CACHES IT FOR FUTURE USE.
'''

from ultralytics import YOLO
import streamlit as st
import os

MODEL_PATH = os.path.join("Model", "yolov8n.pt") # Path to the YOLO model file

@st.cache_resource #Cache the loaded model to avoid reloading on every interaction
def load_model():
    
    model = YOLO(MODEL_PATH)
    return model


model = load_model()
print(model.names)