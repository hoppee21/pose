import numpy as np
import os
import cv2
import pandas as pd

def pick_label(root_dir, N=2, num_idx=3):
    file2label = {}
    annotation_name = 'pose_train.csv'
    label_filename = os.path.join(root_dir, 'annotation', annotation_name)
    df = pd.read_csv(label_filename)

    for i in range(0, len(df)):
        filename = df.loc[i, 'name']
        # print(filename)
        action_type = df.loc[i, 'type']
        label_tmp = df.values[i][num_idx:].astype(np.float64)
        label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)

        assert len(label_tmp) % N == 0

        file2label[filename] = []
        file2label[filename].append(label_tmp)
        file2label[filename].append(action_type)
    return file2label


def save_state_frames(frames, label, save_dir, action_type, video_name):
    for i in range(len(label)-1):
        peak = label[i]

        if i >= len(frames):
            continue
        if peak >= 2:
            indexs = [peak-1, peak, peak+1]
        else:
            indexs = [peak, peak+1]
        if i % 2 == 0:
            sub_s1_save_dir = os.path.join(save_dir, action_type,'state1', video_name , str(i))
            os.makedirs(sub_s1_save_dir, exist_ok=True)
            count = 0

            for frame_index in indexs:
                frame = frames[frame_index]
                save_path = os.path.join(sub_s1_save_dir, str(count) + '.jpg')
                cv2.imwrite(save_path, frame)
                count+=1
        else:
            sub_s1_save_dir = os.path.join(save_dir, action_type, 'state2',video_name , str(i))
            os.makedirs(sub_s1_save_dir, exist_ok=True)
            count = 0

            for frame_index in indexs:
                frame = frames[frame_index]
                save_path = os.path.join(sub_s1_save_dir, str(count) + '.jpg')
                cv2.imwrite(save_path, frame)
                count+=1

def _annotation_transform(root_dir):
    train_type = 'train'
    video_dir = os.path.join(root_dir, 'video', train_type)
    train_save_dir = os.path.join(root_dir, 'extracted')
    save_dir = os.path.join(train_save_dir, train_type)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    file2label = pick_label(root_dir)
    for video_name in file2label:
        video_path = os.path.join(video_dir, video_name)
        print('video_path:', video_path)
        cap = cv2.VideoCapture(video_path)
        frames = []
        if cap.isOpened():
            while True:
                success, frame = cap.read()
                if success is False:
                    break
                frames.append(frame)
        cap.release()

        save_state_frames(frames, file2label[video_name][0], save_dir, file2label[video_name][-1], video_name)



# if __name__ == '__main__':
#     root_dir = '/home/yinhuiwang/New_ver/RepCount_pose'
#     file2label = pick_label(root_dir)
#     print(file2label)