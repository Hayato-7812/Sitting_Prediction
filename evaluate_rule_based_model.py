import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from rule_based_logic import sitting_prediction
import matplotlib.pyplot as plt

def load_preprocessed_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def evaluate_rule_based_model(data):
    y_true = []
    y_pred = []
    for person_data in tqdm(data, desc="Evaluating Rule-based Model"):
        y_true.append(0 if person_data['label'] == 'sitting' else 1)
        prediction = sitting_prediction(person_data)
        y_pred.append(prediction)
    return y_true, y_pred

if __name__ == "__main__":
    input_file = './coco/rule_based_preprocessed_data.json'
    data = load_preprocessed_data(input_file)
    
    y_true, y_pred = evaluate_rule_based_model(data)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("No data to evaluate. Please check the input data and processing logic.")

    print(classification_report(y_true, y_pred, target_names=["sitting", "standing"]))

    # 混同行列を表示して保存
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["sitting", "standing"])
    disp.plot()
    plt.title('Confusion Matrix - Rule-based Model')
    plt.savefig('./coco/confusion_matrix_rule_based.png')
    plt.show()
