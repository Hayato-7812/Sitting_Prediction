import json
from tqdm import tqdm

def load_annotations(annotations_file):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def display_keypoints(annotations, sample_size=5):
    count = 0
    for ann in tqdm(annotations['annotations'], desc="Extracting Keypoints"):
        if ann['num_keypoints'] > 0:
            print(f"Image ID: {ann['image_id']}")
            print(f"Keypoints: {ann['keypoints']}")
            count += 1
        if count >= sample_size:
            break

if __name__ == "__main__":
    annotations_file = './coco/annotations/person_keypoints_val2017.json'
    sample_size = 5  # 表示するサンプル数

    annotations = load_annotations(annotations_file)
    display_keypoints(annotations, sample_size)
