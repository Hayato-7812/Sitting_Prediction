import math
import numpy as np

# インデクスとの対応がわかりずらいので，文字列指定でインデックスを取り出せるようにする
keypoints_dict = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4, "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

def calculate_distance(point1, point2):
    """
    2つの座標点[x, y]の距離を計算するヘルパー関数
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_angle(x1, y1, x2, y2, x3, y3):
    """
    3点からなる角の角度を計算する関数
    """
    a = calculate_distance([x2, y2], [x3, y3])
    b = calculate_distance([x1, y1], [x3, y3])
    c = calculate_distance([x1, y1], [x2, y2])

    if b == 0 or c == 0:
        return 0  # ゼロ除算を防ぐために0を返す

    cos_A = (b**2 + c**2 - a**2) / (2 * b * c)

    # acosの引数が[-1, 1]の範囲内に収まるようにする
    cos_A = max(-1, min(1, cos_A))

    angle_A = math.degrees(math.acos(cos_A))
    return angle_A

def calculate_height(person_data):
    """
    被写体の身長を算出する関数
    """
    left_eye_x = person_data['keypoints'][keypoints_dict["left_eye"]][0]
    left_eye_y = person_data['keypoints'][keypoints_dict["left_eye"]][1]
    right_eye_x = person_data['keypoints'][keypoints_dict["right_eye"]][0]
    right_eye_y = person_data['keypoints'][keypoints_dict["right_eye"]][1]

    nose_x = person_data['keypoints'][keypoints_dict["nose"]][0]
    nose_y = person_data['keypoints'][keypoints_dict["nose"]][1]

    left_shoulder_x = person_data['keypoints'][keypoints_dict["left_shoulder"]][0]
    left_shoulder_y = person_data['keypoints'][keypoints_dict["left_shoulder"]][1]
    right_shoulder_x = person_data['keypoints'][keypoints_dict["right_shoulder"]][0]
    right_shoulder_y = person_data['keypoints'][keypoints_dict["right_shoulder"]][1]

    left_hip_x = person_data['keypoints'][keypoints_dict["left_hip"]][0]
    left_hip_y = person_data['keypoints'][keypoints_dict["left_hip"]][1]
    right_hip_x = person_data['keypoints'][keypoints_dict["right_hip"]][0]
    right_hip_y = person_data['keypoints'][keypoints_dict["right_hip"]][1]

    left_knee_x = person_data['keypoints'][keypoints_dict["left_knee"]][0]
    left_knee_y = person_data['keypoints'][keypoints_dict["left_knee"]][1]
    right_knee_x = person_data['keypoints'][keypoints_dict["right_knee"]][0]
    right_knee_y = person_data['keypoints'][keypoints_dict["right_knee"]][1]

    left_ankle_x = person_data['keypoints'][keypoints_dict["left_ankle"]][0]
    left_ankle_y = person_data['keypoints'][keypoints_dict["left_ankle"]][1]
    right_ankle_x = person_data['keypoints'][keypoints_dict["right_ankle"]][0]
    right_ankle_y = person_data['keypoints'][keypoints_dict["right_ankle"]][1]

    distances = {
        "eye_nose": calculate_distance([(left_eye_x + right_eye_x) / 2, (left_eye_y + right_eye_y) / 2], [nose_x, nose_y]),
        "nose_shoulder": calculate_distance([nose_x, nose_y], [(left_shoulder_x + right_shoulder_x) / 2, (left_shoulder_y + right_shoulder_y) / 2]),
        "shoulder_hip": calculate_distance([(left_shoulder_x + right_shoulder_x) / 2, (left_shoulder_y + right_shoulder_y) / 2], [(left_hip_x + right_hip_x) / 2, (left_hip_y + right_hip_y) / 2]),
        "hip_knee": calculate_distance([(left_hip_x + right_hip_x) / 2, (left_hip_y + right_hip_y) / 2], [(left_knee_x + right_knee_x) / 2, (left_knee_y + right_knee_y) / 2]),
        "knee_ankle": calculate_distance([(left_knee_x + right_knee_x) / 2, (left_knee_y + right_knee_y) / 2], [(left_ankle_x + right_ankle_x) / 2, (left_ankle_y + right_ankle_y) / 2])
    }

    total_height = sum(distances.values())
    return total_height

def body_orientation_prediction(person_data):
    """
    被写体の身体の向きを判定する関数
    - front: 正面を向いている
    - side: 横を向いている
    - unknown: それ以外
    """
    left_eye_x = person_data['keypoints'][keypoints_dict["left_eye"]][0]
    right_eye_x = person_data['keypoints'][keypoints_dict["right_eye"]][0]
    left_ear_x = person_data['keypoints'][keypoints_dict["left_ear"]][0]
    right_ear_x = person_data['keypoints'][keypoints_dict["right_ear"]][0]

    left_shoulder_x = person_data['keypoints'][keypoints_dict["left_shoulder"]][0]
    right_shoulder_x = person_data['keypoints'][keypoints_dict["right_shoulder"]][0]
    left_hip_x = person_data['keypoints'][keypoints_dict["left_hip"]][0]
    right_hip_x = person_data['keypoints'][keypoints_dict["right_hip"]][0]
    left_knee_x = person_data['keypoints'][keypoints_dict["left_knee"]][0]
    right_knee_x = person_data['keypoints'][keypoints_dict["right_knee"]][0]
    right_eye_y = person_data['keypoints'][keypoints_dict["right_eye"]][1]
    left_eye_y = person_data['keypoints'][keypoints_dict["left_eye"]][1]
    right_knee_y = person_data['keypoints'][keypoints_dict["right_knee"]][1]
    left_knee_y = person_data['keypoints'][keypoints_dict["left_knee"]][1]
    right_ankle_x = person_data['keypoints'][keypoints_dict["right_ankle"]][0]
    right_ankle_y = person_data['keypoints'][keypoints_dict["right_ankle"]][1]
    left_ankle_x = person_data['keypoints'][keypoints_dict["left_ankle"]][0]
    left_ankle_y = person_data['keypoints'][keypoints_dict["left_ankle"]][1]

    def is_front_facing():
        face_orientation = (min(right_ear_x, left_ear_x) < min(right_eye_x, left_eye_x) and max(right_eye_x, left_eye_x) < max(right_ear_x, left_ear_x))
        hip_x = (left_hip_x + right_hip_x) / 2
        knee_x = (left_knee_x + right_knee_x) / 2
        shoulder_orientation = min(left_shoulder_x, right_shoulder_x) < hip_x and hip_x < max(left_shoulder_x, right_shoulder_x)
        hip_orientation = min(left_shoulder_x, right_shoulder_x) < knee_x and knee_x < max(left_shoulder_x, right_shoulder_x)
        return face_orientation and shoulder_orientation and hip_orientation

    def is_side_facing(ratio=0.08):
        eye_distance = calculate_distance([right_eye_x, right_eye_y], [left_eye_x, left_eye_y])
        lower_knee_distance = (calculate_distance([right_knee_x, right_knee_y], [right_ankle_x, right_ankle_y]) +
                               calculate_distance([left_knee_x, left_knee_y], [left_ankle_x, left_ankle_y])) / 2
        return eye_distance < lower_knee_distance * ratio

    if is_front_facing():
        return "front"
    elif is_side_facing():
        return "side"
    else:
        return "unknown"

def compare_hip_and_knee(person_data):
    """
    腰の高さが膝よりも高い位置にあるかどうかを基準に判定する関数
    - 0: 座っている
    - 1: 立っている
    """
    left_hip_y = person_data['keypoints'][keypoints_dict["left_hip"]][1]
    right_hip_y = person_data['keypoints'][keypoints_dict["right_hip"]][1]
    left_knee_y = person_data['keypoints'][keypoints_dict["left_knee"]][1]
    right_knee_y = person_data['keypoints'][keypoints_dict["right_knee"]][1]

    hip_y = (left_hip_y + right_hip_y) / 2
    knee_y = max(left_knee_y, right_knee_y)

    if hip_y >= knee_y:
        return 0  # 座っている
    else:
        return 1  # 立っている

def judge_hip_and_knees_bent(person_data, threshold=262):
    """
    肩、腰、膝と腰、膝、足からなる角の角度の合計値が閾値以上であるかどうかを基準に判定する関数
    - 0: 座っている
    - 1: 立っている
    """
    left_shoulder_x = person_data['keypoints'][keypoints_dict["left_shoulder"]][0]
    left_shoulder_y = person_data['keypoints'][keypoints_dict["left_shoulder"]][1]
    right_shoulder_x = person_data['keypoints'][keypoints_dict["right_shoulder"]][0]
    right_shoulder_y = person_data['keypoints'][keypoints_dict["right_shoulder"]][1]

    left_hip_x = person_data['keypoints'][keypoints_dict["left_hip"]][0]
    left_hip_y = person_data['keypoints'][keypoints_dict["left_hip"]][1]
    right_hip_x = person_data['keypoints'][keypoints_dict["right_hip"]][0]
    right_hip_y = person_data['keypoints'][keypoints_dict["right_hip"]][1]

    left_knee_x = person_data['keypoints'][keypoints_dict["left_knee"]][0]
    left_knee_y = person_data['keypoints'][keypoints_dict["left_knee"]][1]
    right_knee_x = person_data['keypoints'][keypoints_dict["right_knee"]][0]
    right_knee_y = person_data['keypoints'][keypoints_dict["right_knee"]][1]

    left_ankle_x = person_data['keypoints'][keypoints_dict["left_ankle"]][0]
    left_ankle_y = person_data['keypoints'][keypoints_dict["left_ankle"]][1]
    right_ankle_x = person_data['keypoints'][keypoints_dict["right_ankle"]][0]
    right_ankle_y = person_data['keypoints'][keypoints_dict["right_ankle"]][1]

    angle_right_hip = calculate_angle(right_shoulder_x, right_shoulder_y, right_hip_x, right_hip_y, right_knee_x, right_knee_y)
    angle_left_hip = calculate_angle(left_shoulder_x, left_shoulder_y, left_hip_x, left_hip_y, left_knee_x, left_knee_y)
    angle_right_knee = calculate_angle(right_hip_x, right_hip_y, right_knee_x, right_knee_y, right_ankle_x, right_ankle_y)
    angle_left_knee = calculate_angle(left_hip_x, left_hip_y, left_knee_x, left_knee_y, left_ankle_x, left_ankle_y)

    angle_hip = (angle_right_hip + angle_left_hip) / 2
    angle_knee = (angle_right_knee + angle_left_knee) / 2

    angle = angle_hip + angle_knee

    if angle < threshold:
        return 0  # 座っている
    else:
        return 1  # 立っている

def compare_bbox_and_hip2knee_width_rate(person_data, threshold=25):
    """
    bboxの幅に対して腰膝間のx軸距離の割合が閾値以上であるかどうかを基準に判定する関数
    - 0: 座っている
    - 1: 立っている
    """
    left_hip_x = person_data['keypoints'][keypoints_dict["left_hip"]][0]
    right_hip_x = person_data['keypoints'][keypoints_dict["right_hip"]][0]
    left_knee_x = person_data['keypoints'][keypoints_dict["left_knee"]][0]
    right_knee_x = person_data['keypoints'][keypoints_dict["right_knee"]][0]

    bbox_width = person_data['bbox'][2]
    keypoints_diff = abs(((left_hip_x + right_hip_x) / 2) - ((left_knee_x + right_knee_x) / 2))

    result = (keypoints_diff / bbox_width) * 100

    if result >= threshold:
        return 0  # 座っている
    else:
        return 1  # 立っている

def compare_height_and_lower_body_height(person_data, threshold=0.47):
    """
    身長に対して下半身の高さ（腰足間のy軸距離）の割合が閾値以上であるかどうかを基準に判定する関数
    - 0: 座っている
    - 1: 立っている
    """
    left_hip_y = person_data['keypoints'][keypoints_dict["left_hip"]][1]
    right_hip_y = person_data['keypoints'][keypoints_dict["right_hip"]][1]
    left_ankle_y = person_data['keypoints'][keypoints_dict["left_ankle"]][1]
    right_ankle_y = person_data['keypoints'][keypoints_dict["right_ankle"]][1]

    lower_body_height = abs((left_hip_y + right_hip_y) / 2 - max(left_ankle_y, right_ankle_y))
    total_height = calculate_height(person_data)

    if total_height == 0:  # 身長がゼロの場合のチェック
        return 0  # 座っていると判定

    ratio = lower_body_height / total_height

    if ratio <= threshold:
        return 0  # 座っている
    else:
        return 1  # 立っている

def compare_height_and_hip2knee_height(person_data, threshold=0.20):
    """
    身長に対して腰膝間の高さ（y軸距離）の割合が閾値以上であるかどうかを基準に判定する関数
    - 0: 座っている
    - 1: 立っている
    """
    left_hip_y = person_data['keypoints'][keypoints_dict["left_hip"]][1]
    right_hip_y = person_data['keypoints'][keypoints_dict["right_hip"]][1]
    left_knee_y = person_data['keypoints'][keypoints_dict["left_knee"]][1]
    right_knee_y = person_data['keypoints'][keypoints_dict["right_knee"]][1]

    distance_hip2knee = abs((left_hip_y + right_hip_y) / 2 - max(left_knee_y, right_knee_y))
    total_height = calculate_height(person_data)

    if total_height == 0:  # 身長がゼロの場合のチェック
        return 0  # 座っていると判定

    ratio = distance_hip2knee / total_height

    if ratio <= threshold:
        return 0  # 座っている
    else:
        return 1  # 立っている

def sitting_prediction(person_data):
    """
    複数の基準を統合して、人物が座っているか立っているかを判定する関数
    - 0: 座っている
    - 1: 立っている
    """
    scores = [
        compare_hip_and_knee(person_data),
        judge_hip_and_knees_bent(person_data),
        compare_height_and_lower_body_height(person_data),
        compare_height_and_hip2knee_height(person_data)
    ]
    
    body_orientation = body_orientation_prediction(person_data)
    if body_orientation == "side":
        scores.append(compare_bbox_and_hip2knee_width_rate(person_data))
    
    return 0 if scores.count(0) > scores.count(1) else 1
