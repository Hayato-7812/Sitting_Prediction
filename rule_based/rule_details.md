## ルールと判断基準

### 1. 距離計算 (`calculate_distance`)

2つの座標点 `[x, y]` の距離を計算します。

```python
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
```

### 2. 角度計算 (`calculate_angle`)

3点からなる角の角度を計算します。

```python
def calculate_angle(x1, y1, x2, y2, x3, y3):
    a = calculate_distance([x2, y2], [x3, y3])
    b = calculate_distance([x1, y1], [x3, y3])
    c = calculate_distance([x1, y1], [x2, y2])

    cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
    angle_A = math.degrees(math.acos(cos_A))
    return angle_A
```

### 3. 身体の向き判定 (`body_orientation_prediction`)

被写体の身体の向きを判定します。正面 (`front`)、横向き (`side`)、それ以外 (`unknown`) の3つの状態を返します。

```python
def body_orientation_prediction(person_data):
    def is_front_facing():
        face_orientation = (min(right_ear_x, left_ear_x) < min(right_eye_x, left_eye_x) and max(right_eye_x, left_eye_x) < max(right_ear_x, left_ear_x))
        waist_x = (left_waist_x + right_waist_x) / 2
        knee_x = (left_knee_x + right_knee_x) / 2
        shoulder_orientation = min(left_shoulder_x, right_shoulder_x) < waist_x and waist_x < max(left_shoulder_x, right_shoulder_x)
        waist_orientation = min(left_shoulder_x, right_shoulder_x) < knee_x and knee_x < max(left_shoulder_x, right_shoulder_x)
        return face_orientation and shoulder_orientation and waist_orientation

    def is_side_facing(ratio=0.08):
        eye_distance = calculate_distance([right_eye_x, right_eye_x], [left_eye_x, left_eye_x])
        lower_knee_distance = (calculate_distance([right_knee_x, right_knee_x], [right_shoulder_x, right_shoulder_x]) + calculate_distance([left_knee_x, left_knee_x], [left_shoulder_x, left_shoulder_x])) / 2
        return eye_distance < lower_knee_distance * ratio

    if is_front_facing():
        return "front"
    elif is_side_facing():
        return "side"
    else:
        return "unknown"
```

### 4. 腰と膝の高さの比較 (`compare_waist_and_knee`)

腰の高さが膝よりも高い位置にあるかどうかを基準に判定します。

- 0: 座っている
- 1: 立っている

```python
def compare_waist_and_knee(person_data):
    left_waist_y = person_data['keypoints'][keypoints_dict["左腰"]][1]
    right_waist_y = person_data['keypoints'][keypoints_dict["右腰"]][1]
    left_knee_y = person_data['keypoints'][keypoints_dict["左膝"]][1]
    right_knee_y = person_data['keypoints'][keypoints_dict["右膝"]][1]

    waist_y = (left_waist_y + right_waist_y) / 2
    knee_y = max(left_knee_y, right_knee_y)

    if waist_y >= knee_y:
        return 0  # 座っている
    else:
        return 1  # 立っている
```

### 5. 腰と膝の角度の判定 (`judge_waist_and_knees_bent`)

肩、腰、膝と腰、膝、足からなる角の角度の合計値が閾値以上であるかどうかを基準に判定します。

- 0: 座っている
- 1: 立っている

```python
def judge_waist_and_knees_bent(person_data, threshold=262):
    left_shoulder_x = person_data['keypoints'][keypoints_dict["左肩"]][0]
    left_shoulder_y = person_data['keypoints'][keypoints_dict["左肩"]][1]
    right_shoulder_x = person_data['keypoints'][keypoints_dict["右肩"]][0]
    right_shoulder_y = person_data['keypoints'][keypoints_dict["右肩"]][1]

    left_waist_x = person_data['keypoints'][keypoints_dict["左腰"]][0]
    left_waist_y = person_data['keypoints'][keypoints_dict["左腰"]][1]
    right_waist_x = person_data['keypoints'][keypoints_dict["右腰"]][0]
    right_waist_y = person_data['keypoints'][keypoints_dict["右腰"]][1]

    left_knee_x = person_data['keypoints'][keypoints_dict["左膝"]][0]
    left_knee_y = person_data['keypoints'][keypoints_dict["左膝"]][1]
    right_knee_x = person_data['keypoints'][keypoints_dict["右膝"]][0]
    right_knee_y = person_data['keypoints'][keypoints_dict["右膝"]][1]

    left_foot_x = person_data['keypoints'][keypoints_dict["左足"]][0]
    left_foot_y = person_data['keypoints'][keypoints_dict["左足"]][1]
    right_foot_x = person_data['keypoints'][keypoints_dict["右足"]][0]
    right_foot_y = person_data['keypoints'][keypoints_dict["右足"]][1]

    angle_right_waist = calculate_angle(right_shoulder_x, right_shoulder_y, right_waist_x, right_waist_y, right_knee_x, right_knee_y)
    angle_left_waist = calculate_angle(left_shoulder_x, left_shoulder_y, left_waist_x, left_waist_y, left_knee_x, left_knee_y)
    angle_right_knee = calculate_angle(right_waist_x, right_waist_y, right_knee_x, right_knee_y, right_foot_x, right_foot_y)
    angle_left_knee = calculate_angle(left_waist_x, left_waist_y, left_knee_x, left_knee_y, left_foot_x, left_foot_y)

    angle_waist = (angle_right_waist + angle_left_waist) / 2
    angle_knee = (angle_right_knee + angle_left_knee) / 2

    angle = angle_waist + angle_knee

    if angle < threshold:
        return 0  # 座っている
    else:
        return 1  # 立っている
```

### 6. 腰と膝の幅の割合の比較 (`compare_bbox_and_waist2knee_width_rate`)

bboxの幅に対して腰膝間のx軸距離の割合が閾値以上であるかどうかを基準に判定します。

- 0: 座っている
- 1: 立っている

```python
def compare_bbox_and_waist2knee_width_rate(person_data, threshold=25):
    left_waist_x = person_data['keypoints'][keypoints_dict["左腰"]][0]
    right_waist_x = person_data['keypoints'][keypoints_dict["右腰"]][0]
    left_knee_x = person_data['keypoints'][keypoints_dict["左膝"]][0]
    right_knee_x = person_data['keypoints'][keypoints_dict["右膝"]][0]

    bbox_width = person_data['bbox']['w']
    keypoints_diff = abs(((left_waist_x + right_waist_x) / 2) - ((left_knee_x + right_knee_x) / 2))

    result = (keypoints_diff / bbox_width) * 100

    if result >= threshold:
        return 0  # 座っている
    else:
        return 1  # 立っている
```

### 7. 身長計算 (`calculate_height`)

被写体の身長を計算します。

```python
def calculate_height(person_data):
    left_eye_x = person_data['keypoints'][keypoints_dict["左目"]][0]
    left_eye_y = person_data['key

points'][keypoints_dict["左目"]][1]
    right_eye_x = person_data['keypoints'][keypoints_dict["右目"]][0]
    right_eye_y = person_data['keypoints'][keypoints_dict["右目"]][1]

    nose_x = person_data['keypoints'][keypoints_dict["鼻"]][0]
    nose_y = person_data['keypoints'][keypoints_dict["鼻"]][1]

    left_shoulder_x = person_data['keypoints'][keypoints_dict["左肩"]][0]
    left_shoulder_y = person_data['keypoints'][keypoints_dict["左肩"]][1]
    right_shoulder_x = person_data['keypoints'][keypoints_dict["右肩"]][0]
    right_shoulder_y = person_data['keypoints'][keypoints_dict["右肩"]][1]

    left_waist_x = person_data['keypoints'][keypoints_dict["左腰"]][0]
    left_waist_y = person_data['keypoints'][keypoints_dict["左腰"]][1]
    right_waist_x = person_data['keypoints'][keypoints_dict["右腰"]][0]
    right_waist_y = person_data['keypoints'][keypoints_dict["右腰"]][1]

    left_knee_x = person_data['keypoints'][keypoints_dict["左膝"]][0]
    left_knee_y = person_data['keypoints'][keypoints_dict["左膝"]][1]
    right_knee_x = person_data['keypoints'][keypoints_dict["右膝"]][0]
    right_knee_y = person_data['keypoints'][keypoints_dict["右膝"]][1]

    left_foot_x = person_data['keypoints'][keypoints_dict["左足"]][0]
    left_foot_y = person_data['keypoints'][keypoints_dict["左足"]][1]
    right_foot_x = person_data['keypoints'][keypoints_dict["右足"]][0]
    right_foot_y = person_data['keypoints'][keypoints_dict["右足"]][1]

    distances = {
        "eye_nose": calculate_distance([(left_eye_x + right_eye_x) / 2, (left_eye_y + right_eye_y) / 2], [nose_x, nose_y]),
        "nose_shoulder": calculate_distance([nose_x, nose_y], [(left_shoulder_x + right_shoulder_x) / 2, (left_shoulder_y + right_shoulder_y) / 2]),
        "shoulder_waist": calculate_distance([(left_shoulder_x + right_shoulder_x) / 2, (left_shoulder_y + right_shoulder_y) / 2], [(left_waist_x + right_waist_x) / 2, (left_waist_y + right_waist_y) / 2]),
        "waist_knee": calculate_distance([(left_waist_x + right_waist_x) / 2, (left_waist_y + right_waist_y) / 2], [(left_knee_x + right_knee_x) / 2, (left_knee_y + right_knee_y) / 2]),
        "knee_foot": calculate_distance([(left_knee_x + right_knee_x) / 2, (left_knee_y + right_knee_y) / 2], [(left_foot_x + right_foot_x) / 2, (left_foot_y + right_foot_y) / 2])
    }

    total_height = sum(distances.values())
    return total_height
```

### 8. 下半身の高さの割合の比較 (`compare_height_and_lower_body_height`)

身長に対して下半身の高さ（腰足間のy軸距離）の割合が閾値以上であるかどうかを基準に判定します。

- 0: 座っている
- 1: 立っている

```python
def compare_height_and_lower_body_height(person_data, threshold=0.47):
    left_waist_y = person_data['keypoints'][keypoints_dict["左腰"]][1]
    right_waist_y = person_data['keypoints'][keypoints_dict["右腰"]][1]
    left_foot_y = person_data['keypoints'][keypoints_dict["左足"]][1]
    right_foot_y = person_data['keypoints'][keypoints_dict["右足"]][1]

    lower_body_height = abs((left_waist_y + right_waist_y) / 2 - max(left_foot_y, right_foot_y))
    total_height = calculate_height(person_data)

    ratio = lower_body_height / total_height

    if ratio <= threshold:
        return 0  # 座っている
    else:
        return 1  # 立っている
```

### 9. 腰膝間の高さの割合の比較 (`compare_height_and_waist2knee_height`)

身長に対して腰膝間の高さ（y軸距離）の割合が閾値以上であるかどうかを基準に判定します。

- 0: 座っている
- 1: 立っている

```python
def compare_height_and_waist2knee_height(person_data, threshold=0.20):
    left_waist_y = person_data['keypoints'][keypoints_dict["左腰"]][1]
    right_waist_y = person_data['keypoints'][keypoints_dict["右腰"]][1]
    left_knee_y = person_data['keypoints'][keypoints_dict["左膝"]][1]
    right_knee_y = person_data['keypoints'][keypoints_dict["右膝"]][1]

    distance_waist2knee = abs((left_waist_y + right_waist_y) / 2 - max(left_knee_y, right_knee_y))
    total_height = calculate_height(person_data)

    ratio = distance_waist2knee / total_height

    if ratio <= threshold:
        return 0  # 座っている
    else:
        return 1  # 立っている
```

### 10. 総合判定 (`sitting_prediction`)

複数の基準を統合して、人物が座っているか立っているかを判定します。

- 0: 座っている
- 1: 立っている

```python
def sitting_prediction(person_data):
    body_orientation = body_orientation_prediction(person_data)
    if body_orientation == "side":
        return compare_bbox_and_waist2knee_width_rate(person_data)
    
    scores = [
        compare_waist_and_knee(person_data),
        judge_waist_and_knees_bent(person_data),
        compare_height_and_lower_body_height(person_data),
        compare_height_and_waist2knee_height(person_data)
    ]
    
    return 0 if scores.count(0) > scores.count(1) else 1
```

### 11. データの前処理とルールベースのラベリング (`preprocess_data_with_rule_based`)

データを前処理し、ルールベースの手法を適用してラベルを付けます。

```python
def preprocess_data_with_rule_based(input_file, output_file):
    with open(input_file, 'r') as f:
        annotations = json.load(f)
    
    data = []
    for ann in tqdm(annotations['annotations'], desc="Processing Data with Rule-based Method"):
        if ann['num_keypoints'] > 0:
            person_data = {
                'bbox': {'w': ann['bbox'][2]},
                'keypoints': np.array(ann['keypoints']).reshape(-1, 3)[:, :2].tolist()  # xy座標のみ
            }
            label = sitting_prediction(person_data)
            data.append({
                'keypoints': person_data['keypoints'],
                'label': 'sitting' if label == 0 else 'standing'
            })
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
```

---

以上が、設定されているルールと判断基準の詳細です。