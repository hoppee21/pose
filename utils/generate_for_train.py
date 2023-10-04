import os
import cv2
import csv
import numpy as np
from mmpose.apis import MMPoseInferencer


def init_mmpose():

    inference = MMPoseInferencer(

    pose2d= '/home/yihuiwang/anaconda3/envs/mypose/lib/python3.9/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-384x288.py',

    pose2d_weights='/home/yihuiwang/New_ver/utils/td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth',

    device= 'cuda:1'
)
    
    return inference

def mmpose_process(frame,model,writter, path, action_type):

    result_generator = model(frame)
    result = next(result_generator)
    # keypoints = np.array(result['predictions'][0][0]['keypoints'], dtype=np.float32)
    keypoints = result['predictions'][0][0]['keypoints']
    score = result['predictions'][0][0]['keypoint_scores']

    for i in range(len(keypoints)):
        keypoints[i].append(score[i])
    keypoints = np.array(keypoints, dtype=np.float32)
    assert keypoints.shape == (17, 3)
    'Unexpected keypoint shape: {}'.format(keypoints.shape)
    writter.writerow([path, action_type] + keypoints.flatten().astype(str).tolist())
    
def frame_process(action_type,state,folder,model,writter):
    
    for videos in os.listdir(folder):
        
        period_dir = os.path.join(folder, videos)
        if '.DS_Store' in period_dir:
            continue

        for period in os.listdir(period_dir):
            video_dir = os.path.join(period_dir, period)

            for frame_path in os.listdir(video_dir):
                if '.jpg' not in frame_path:
                    continue
                frame = os.path.join(video_dir,frame_path)
                print(frame)
                base_path = os.path.join('train', action_type, state, period, frame_path)
                input_frame = cv2.imread(frame)
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                mmpose_process(input_frame,model,writter,base_path,action_type)

                    
def _generate_for_train(root_dir):

    data_folder = os.path.join(root_dir, 'extracted')
    out_csv_dir = os.path.join(root_dir, 'annotation_pose')
    os.makedirs(out_csv_dir, exist_ok=True)

    mmpose_model = init_mmpose()

    for train_type in os.listdir(data_folder):
        if '.DS_Store' in train_type:
            continue
        out_csv_path = os.path.join(out_csv_dir, train_type) + '.csv'
        with open(out_csv_path, 'w') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            sub_train_folder = os.path.join(data_folder, train_type)

            for action_type in os.listdir(sub_train_folder):
                sub_sub_folder = os.path.join(sub_train_folder, action_type)
                print(action_type)
                if '.DS_Store' in action_type:
                    continue
                for state in os.listdir(sub_sub_folder):
                    sub_sub_sub_folder = os.path.join(sub_sub_folder, state)
                    if '.DS_Store' in state:
                        continue
                    frame_process(action_type, state, sub_sub_sub_folder,mmpose_model, csv_out_writer)