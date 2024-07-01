import json
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from tqdm import tqdm

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
        for key in [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]:
            keypoints_flatten.extend(keypoints[key])
        X.append(keypoints_flatten)
        y.append(0 if item['label'] == 'sitting' else 1)  # 'sitting' -> 0, 'standing' -> 1
    return np.array(X), np.array(y)

if __name__ == "__main__":
    input_file = './coco/preprocessed_data.json'
    model_output_file = './coco/svm_model.pkl'

    data = load_preprocessed_data(input_file)
    X, y = prepare_dataset(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = svm.SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_output_file)
