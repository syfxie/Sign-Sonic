import os
from tqdm import tqdm
import numpy as np
import cv2 as cv
import mediapipe as mp

NUM_FACE_LANDMARKS = 468
NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33
TOTAL_LANDMARKS = NUM_FACE_LANDMARKS + NUM_HAND_LANDMARKS * 2 + NUM_POSE_LANDMARKS


def extract_frame_keypoints(all_landmarks):
    left_hand_arr = np.zeros(NUM_HAND_LANDMARKS * 3)
    right_hand_arr = np.zeros(NUM_HAND_LANDMARKS * 3)
    face_arr = np.zeros(NUM_FACE_LANDMARKS * 3)
    pose_arr = np.zeros(NUM_POSE_LANDMARKS * 4)

    # left hand
    if all_landmarks.left_hand_landmarks:
        left_hand_arr = np.array(
            [[pos.x, pos.y, pos.z] for pos in all_landmarks.left_hand_landmarks.landmark]).flatten()
        # print('left', left_hand_arr.shape)

    # right hand
    if all_landmarks.right_hand_landmarks:
        right_hand_arr = np.array(
            [[pos.x, pos.y, pos.z] for pos in all_landmarks.right_hand_landmarks.landmark]).flatten()
        # print('right', right_hand_arr.shape)

    # face
    if all_landmarks.face_landmarks:
        face_arr = np.array([[pos.x, pos.y, pos.z] for pos in all_landmarks.face_landmarks.landmark]).flatten()
        # print(face_arr.shape)

    # pose
    if all_landmarks.pose_landmarks:
        pose_arr = np.array(
            [[pos.x, pos.y, pos.z, pos.visibility] for pos in all_landmarks.pose_landmarks.landmark]).flatten()
        # print(pose_arr.shape)

    return np.concatenate([left_hand_arr, right_hand_arr, face_arr, pose_arr])


def extract_video_keypoints(video_path, start=1, end=-1):
    cap = cv.VideoCapture(video_path)

    if start < 1 or start > int(cap.get(cv.CAP_PROP_FRAME_COUNT)):
        start = 1

    if end < min(0, start):
        end = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    num_frames = end - start + 1
    keypoints_arr_size = NUM_HAND_LANDMARKS * 3 + NUM_HAND_LANDMARKS * 3 + NUM_FACE_LANDMARKS * 3 + NUM_POSE_LANDMARKS * 4
    video_keypoints = np.zeros((num_frames, keypoints_arr_size))

    frame_index = 1

    # Set mediapipe model
    with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic_model:
        while cap.isOpened() and frame_index <= end:

            # Read feed
            success, frame = cap.read()

            if not success:
                print("Failed to read frame.")
                break

            if frame_index >= start:
                frame.flags.writeable = False
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame_landmarks = holistic_model.process(frame)
                frame_keypoints_arr = extract_frame_keypoints(frame_landmarks)
                video_keypoints[frame_index - start] = frame_keypoints_arr

            frame_index += 1

        cap.release()
        cv.destroyAllWindows()
    return video_keypoints


def save_keypoints(data, dest):
    os.makedirs(dest, exist_ok=True)

    for i in tqdm(range(len(data)), ncols=100):
        # for i in range(100):
        video_id = data[i]['video_id']
        npy_path = os.path.join(dest, f'{video_id}.npy')

        if not os.path.exists(npy_path):
            video_path = data[i]['video_path']
            start = data[i]['frame_start']
            end = data[i]['frame_end']

            try:
                video_keypoints = extract_video_keypoints(video_path, start, end)
                print(npy_path)
                np.save(npy_path, video_keypoints)

            except Exception as e:
                print(f"Failed on video at {video_path}:{e}")
                continue


if __name__ == '__main__':
    save_keypoints([], './Keypoints')

