import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_preprocessed_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def prepare_dataset(data):
    X = []
    y = []
    for item in tqdm(data, desc="Preparing Dataset"):
        keypoints = np.array(item['keypoints']).flatten()
        X.append(keypoints)
        y.append(item['label'])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    ml_data_file = './coco/preprocessed_data.json'
    model_file = './coco/svm_model.pkl'
    ml_confusion_matrix_output_file = './coco/ml_confusion_matrix.png'

    data = load_preprocessed_data(ml_data_file)
    X, y = prepare_dataset(data)
    
    model = joblib.load(model_file)
    y_pred = model.predict(X)
    
    cm = confusion_matrix(y, y_pred, labels=["sitting", "standing"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["sitting", "standing"])
    disp.plot()
    plt.title('Confusion Matrix (Machine Learning)')
    plt.savefig(ml_confusion_matrix_output_file)
    plt.show()
