import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

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
    model_file = './coco/svm_model.pkl'
    confusion_matrix_output_file = './coco/confusion_matrix.png'
    svm_classification_output_file = './coco/svm_classification.png'
    pca_components_output_file = './coco/pca_components.png'

    data = load_preprocessed_data(input_file)
    X, y = prepare_dataset(data)
    
    model = joblib.load(model_file)
    
    y_pred = model.predict(X)
    
    # 混同行列を表示して保存
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["sitting", "standing"])
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(confusion_matrix_output_file)
    plt.show()
    
    # PCAによる次元削減
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    # 関節点のラベル
    joint_labels = [
        'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder',
        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee',
        'R_Knee', 'L_Ankle', 'R_Ankle'
    ]
    feature_labels = []
    for label in joint_labels:
        feature_labels.append(f'{label}_x')
        feature_labels.append(f'{label}_y')
    
    # PCAの主成分を表示
    plt.figure(figsize=(10, 6))
    x = np.arange(len(feature_labels))
    width = 0.35

    # 正しく特徴数に対応するように
    if len(pca.components_[0]) == len(feature_labels):
        plt.bar(x - width/2, pca.components_[0], width, label=f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)')
        plt.bar(x + width/2, pca.components_[1], width, label=f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)')
    else:
        print(f'Warning: PCA components and feature labels length mismatch. PCA components: {len(pca.components_[0])}, feature labels: {len(feature_labels)}')
        plt.bar(x - width/2, pca.components_[0][:len(feature_labels)], width, label=f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)')
        plt.bar(x + width/2, pca.components_[1][:len(feature_labels)], width, label=f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)')

    plt.xlabel('Features')
    plt.ylabel('Principal Component Value')
    plt.title('Principal Component Analysis')
    plt.xticks(x, feature_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pca_components_output_file)
    plt.show()
    
    # SVMによるクラス分けを表示
    plt.figure(figsize=(8, 6))
    for label, color in zip([0, 1], ['red', 'blue']):  # 'sitting' -> 0, 'standing' -> 1
        indices = np.where(y == label)
        plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], c=color, label='sitting' if label == 0 else 'standing', alpha=0.5)
    
    plt.legend()
    plt.title('SVM Classification with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(svm_classification_output_file)
    plt.show()
