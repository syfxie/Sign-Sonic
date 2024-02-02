import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def preprocess_data(data, target_frames=50, step=10, padding_value=0, keypoints_dir='./Keypoints/'):
    data_sequence = []
    label_sequence = []
    
    for entry in data:
        video_id = entry['video_id']
        npy_path = keypoints_dir + video_id + '.npy'
    
        # if the numpy array of keypoints exist, record the label and load the array
        if os.path.exists(npy_path):
            print(npy_path)
            npy_arr = np.load(npy_path)
            num_frames = npy_arr.shape[0]
    
            # not enough frames
            if num_frames < target_frames:
                npy_arr = np.pad(npy_arr, ((0, target_frames - num_frames), (0, 0)), mode='constant',
                                 constant_values=padding_value)
                print(npy_arr.shape)
    
            # extra frames
            if num_frames > target_frames:
                for start_index in range(0, num_frames - target_frames + 1, step):
                    end = start_index + target_frames
                    # print(npy_arr[start_index:end].shape)
                    data_sequence.append(npy_arr[start_index:end])
                    label_sequence.append(entry['gloss'])
    
            if num_frames == target_frames:
                print('Exactly {} frames found'.format(target_frames))
    
    print(len(data_sequence) == len(label_sequence))
    print(np.array(data_sequence).shape)
    print(np.array(label_sequence).shape)

    X = np.array(data_sequence)

    label_encoder = LabelEncoder()
    int_labels = label_encoder.fit_transform(np.array(label_sequence))
    y = to_categorical(int_labels)

    return X, y
