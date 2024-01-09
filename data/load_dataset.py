import os
import json
from tqdm import tqdm


def load_data(data_path='./WLASL_v0.3.json', dest_path='./processed_data.json', videos_dir='./VVideos'):
    with open(data_path, 'r') as file:
        json_data = json.load(file)

    processed_data = []

    for i in tqdm(range(len(json_data))):
        gloss = json_data[i]['gloss']
        instances = json_data[i]['instances']

        for instance in instances:
            # print(instance['video_id'])
            video_id = instance['video_id']

            if os.path.exists(os.path.join(videos_dir, f'{video_id}.mp4')):
                video_path = os.path.join(videos_dir, f'{video_id}.mp4')
                frame_start = instance['frame_start']
                frame_end = instance['frame_end']
                # split = instance['split']

                processed_data.append({
                    'gloss': gloss,
                    'video_id': video_id,
                    'video_path': video_path,
                    'frame_start': frame_start,
                    'frame_end': frame_end,
                })

    # write data to csv file
    with open(dest_path, 'w') as file:
        json.dump(processed_data, file, indent=4)