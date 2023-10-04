from mmpose.apis import MMPoseInferencer

inference = MMPoseInferencer(

    pose2d= '/home/yihuiwang/anaconda3/envs/mypose/lib/python3.9/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-384x288.py',

    pose2d_weights='td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth',

    device= 'cuda:0'
)

img = '0.jpg'
result_generator = inference(img,return_datasample=False, out_dir='output')
result = next(result_generator)

keypoints = result['predictions'][0][0]['keypoints']
score = result['predictions'][0][0]['keypoint_scores']

for i in range(len(keypoints)):
    keypoints[i].append(score[i])

print(keypoints)