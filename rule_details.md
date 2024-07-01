# 関数名と機能、判断基準

## 関数一覧

### calculate_distance(point1, point2)
- **機能**: 2つの座標点 [x, y] の距離を計算する。
- **引数**:
  - point1: 最初の点の座標 [x, y]
  - point2: 2番目の点の座標 [x, y]
- **戻り値**: 2点間の距離 (float)

### calculate_angle(x1, y1, x2, y2, x3, y3)
- **機能**: 3点からなる角の角度を計算する。
- **引数**:
  - x1, y1: 最初の点の座標
  - x2, y2: 2番目の点の座標
  - x3, y3: 3番目の点の座標
- **戻り値**: 角度 (float)

### calculate_height(person_data)
- **機能**: 被写体の身長を算出する。
- **引数**:
  - person_data: キーポイントの座標データを含む辞書
- **戻り値**: 身長 (float)

### body_orientation_prediction(person_data)
- **機能**: 被写体の身体の向きを判定する。
  - "front": 正面を向いている
  - "side": 横を向いている
  - "unknown": それ以外
- **引数**:
  - person_data: キーポイントの座標データを含む辞書
- **戻り値**: 身体の向き (str)

### compare_hip_and_knee(person_data)
- **機能**: 腰の高さが膝よりも高い位置にあるかどうかを判定する。
- **引数**:
  - person_data: キーポイントの座標データを含む辞書
- **戻り値**: 0 (座っている) または 1 (立っている)

### judge_hip_and_knees_bent(person_data, threshold=262)
- **機能**: 肩、腰、膝と腰、膝、足からなる角の角度の合計値が閾値以上であるかどうかを判定する。
- **引数**:
  - person_data: キーポイントの座標データを含む辞書
  - threshold: 閾値 (デフォルト: 262)
- **戻り値**: 0 (座っている) または 1 (立っている)

### compare_bbox_and_hip2knee_width_rate(person_data, threshold=25)
- **機能**: bboxの幅に対して腰膝間のx軸距離の割合が閾値以上であるかどうかを判定する。
- **引数**:
  - person_data: キーポイントの座標データを含む辞書
  - threshold: 閾値 (デフォルト: 25)
- **戻り値**: 0 (座っている) または 1 (立っている)

### compare_height_and_lower_body_height(person_data, threshold=0.47)
- **機能**: 身長に対して下半身の高さ（腰足間のy軸距離）の割合が閾値以上であるかどうかを判定する。
- **引数**:
  - person_data: キーポイントの座標データを含む辞書
  - threshold: 閾値 (デフォルト: 0.47)
- **戻り値**: 0 (座っている) または 1 (立っている)

### compare_height_and_hip2knee_height(person_data, threshold=0.20)
- **機能**: 身長に対して腰膝間の高さ（y軸距離）の割合が閾値以上であるかどうかを判定する。
- **引数**:
  - person_data: キーポイントの座標データを含む辞書
  - threshold: 閾値 (デフォルト: 0.20)
- **戻り値**: 0 (座っている) または 1 (立っている)

### sitting_prediction(person_data)
- **機能**: 複数の基準を統合して、人物が座っているか立っているかを判定する。
- **引数**:
  - person_data: キーポイントの座標データを含む辞書
- **戻り値**: 0 (座っている) または 1 (立っている)
