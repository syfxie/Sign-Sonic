import os
import cv2 as cv
import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(image):
    holistic = mp.solutions.holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Detect Keypoints
    result = holistic.process(rgb_image)
    return result


def process_video(video_path, parent_path, num_frames):
    frames_sequence = []
    resized_image, result = None, None
    
    if os.path.isfile(video_path):
        abs_video_path = os.path.abspath(video_path)
        abs_parent_path = os.path.abspath(parent_path)

        if not os.path.isdir(abs_parent_path):
            os.makedirs(abs_parent_path)

        print("Converting video from: %s into frames at %s" % (abs_video_path, abs_parent_path))

        frames_collected = 0
        interval = 20
        count = 0

        frame, result = None, None

        cap = cv.VideoCapture(abs_video_path)
        
        while frames_collected < num_frames:
            ret, frame = cap.read()  # extract frame
            
            if not ret:
                break

            # process and save every one in twenty frames
            if count % interval == 0:
                frame_path = os.path.join(abs_parent_path, "{}.jpeg".format(str(frames_collected)))
                
                print('Keypoints for video {}, frame {}: '.format(video_path,  frames_collected))
                
                if not os.path.exists(frame_path):
                    result = extract_keypoints(frame)

                    # save image to file
                    cv.imwrite(frame_path, frame)

                    # add Keypoints array to array of Keypoints for all frames
                    frames_sequence.append(result)
                frames_collected += 1
            
            count += 1

            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()

        # repeat last frame until the target number of frames are collected
        while result and frame and frames_collected <  num_frames:
            frame_path = os.path.join(abs_parent_path, "{}.jpeg".format(str(frames_collected)))

            if not os.path.exists(frame_path):
                result = extract_keypoints(frame)
                cv.imwrite(frame_path, frame)
                frames_sequence.append(result)
            frames_collected += 1

    return frames_sequence
