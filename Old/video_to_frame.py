import os
import json
import cv2 as cv
import shutil

def video_to_frames(video_path, size=None):
    '''
    Extracts a series of frames from a video
    '''
    cap = cv.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
    
        if ret:
            if size:
                frame = cv.resize(frame, size)
            frames.append(frame)
        else:
            break

    cap.release()

    return frames

def process_image(image, size=(64, 64)):
    """
    Returns the resized image raw and as a normalized numpy array
    """
    resized_image = image.resize(size)
    image_array = np.array(resized_image)

    print("Shape of the resized image:", image_array.shape)

    # Normalize the image array between 0-1
    return image_array / 255.0