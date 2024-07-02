import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rule_based_logic import sitting_prediction

# キーポイントのラベル
keypoints_labels = [
    "1", "2", "3", "4", "5", 
    "6", "7", "8", "9", 
    "10", "11", "12", "13", 
    "14", "15", "16", "17"
]

def load_preprocessed_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def prepare_dataset(data):
    X = []
    y = []
    for item in tqdm(data, desc="Preparing Dataset"):
        keypoints = item['keypoints']
        keypoints_flatten = []
        for point in keypoints:
            keypoints_flatten.extend(point[:2])  # x, y 座標のみを使用
        X.append(keypoints_flatten)
        y.append(0 if item['label'] == 'sitting' else 1)  # 'sitting' -> 0, 'standing' -> 1
    return np.array(X), np.array(y)

def visualize_pose(person_data, ax):
    keypoints = person_data['keypoints']
    
    for i, point in enumerate(keypoints):
        x, y = point[:2]
        if x != 0 and y != 0:
            ax.plot(x, y, 'bo')
            ax.text(x, y, keypoints_labels[i], fontsize=8, ha='right')

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), 
        (5, 6), (5, 7), (6, 8), 
        (7, 9), (8, 10), (11, 12), 
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    for start, end in skeleton:
        if keypoints[start][0] != 0 and keypoints[end][0] != 0:
            x1, y1 = keypoints[start][:2]
            x2, y2 = keypoints[end][:2]
            ax.plot([x1, x2], [y1, y2], 'b-')

    predicted_label = person_data.get('predicted_label', 'unknown')
    actual_label = person_data['label']
    correctness = "Correct" if predicted_label == actual_label else "Incorrect"
    
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f"Predicted: {predicted_label}\nActual: {actual_label}\n{correctness}", fontsize=10)

def visualize_poses(data):
    fig, axes = plt.subplots(1, len(data), figsize=(15, 5))
    if len(data) == 1:
        axes = [axes]
    
    for ax, person_data in zip(axes, data):
        visualize_pose(person_data, ax)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_file = './sample.json'
    
    data = load_preprocessed_data(input_file)
    
    for person_data in data:
        prediction = sitting_prediction(person_data)
        person_data['predicted_label'] = 'sitting' if prediction == 0 else 'standing'
        print(f"Predicted: {'sitting' if prediction == 0 else 'standing'} - Actual: {person_data['label']}")
    
    visualize_poses(data)
