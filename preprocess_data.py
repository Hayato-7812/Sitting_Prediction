import json
import os
import numpy as np
from tqdm import tqdm
import random

# キーポイントのインデクス
keypoints_dict = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

def load_annotations(annotations_file):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def sample_images(annotations, sample_size):
    image_ids = [ann['image_id'] for ann in annotations['annotations'] if ann['num_keypoints'] > 0]
    sampled_image_ids = random.sample(image_ids, sample_size)
    return set(sampled_image_ids)

def extract_keypoints(annotations, sampled_image_ids):
    data = []
    for ann in tqdm(annotations['annotations'], desc="Extracting Keypoints"):
        if ann['image_id'] in sampled_image_ids and ann['num_keypoints'] > 0:
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            label = 'sitting' if keypoints[keypoints_dict["left_hip"], 1] > keypoints[keypoints_dict["left_knee"], 1] else 'standing'
            data.append({
                'bbox': ann['bbox'],  # bbox情報を追加
                'keypoints': keypoints[:, :2].tolist(),  # xy座標
                'label': label
            })
    return data

def save_preprocessed_data(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f)
        
def load_preprocessed_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # annotations_file = './coco/annotations/person_keypoints_train2017.json'
    # output_file = './coco/rule_based_preprocessed_data.json'
    # sample_size = 1000  # サンプル数

    # annotations = load_annotations(annotations_file)
    # sampled_image_ids = sample_images(annotations, sample_size)
    # data = extract_keypoints(annotations, sampled_image_ids)
    # save_preprocessed_data(data, output_file)

    # # デバッグ用にデータの一部を表示  
    input_file = './coco/rule_based_preprocessed_data.json'
    data = load_preprocessed_data(input_file)
    
    # デバッグ用にデータの一部を表示
    print(f"Data sample (first 5 items): {data[:5]}")
