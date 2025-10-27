'''
 Module with utility functions for video processing.
 
'''

import cv2
import numpy as np


'''
 Function to save a list of frames as a video
 frames: list of annotated frames (numpy arrays)
 output_path: path to save the video
 fps: frames per second
 Returns: path of the saved video
'''


def save_video(frames, output_path, fps=24):
    
    if len(frames) == 0: #Safety check
        raise ValueError("No frames to save!")
        
    height, width, _= frames[0].shape  # Get frame dimensions (not storing channel since not needed )

    # Define the codec and create VideoWriter object
    video_codec = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 codec
    video_writer = cv2.VideoWriter(output_path, video_codec, fps, (width, height))

    for frame in frames:
       video_writer.write(frame)  # Write each frame to the video

    video_writer.release()  # Release the VideoWriter
    return output_path
