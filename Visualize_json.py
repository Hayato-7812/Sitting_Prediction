import json
import pprint

def load_and_display_coco_json(json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    print("COCO JSON Structure:")
    print("=====================")
    for key, value in coco_data.items():
        print(f"{key} (type: {type(value).__name__}):")
        
        if isinstance(value, list):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First item: {type(value[0]).__name__}")
                if isinstance(value[0], dict):
                    pprint.pprint(value[0], depth=1, width=80)
                else:
                    print(f"  {value[0]}")
        elif isinstance(value, dict):
            pprint.pprint(value, depth=1, width=80)
        else:
            print(f"  {value}")

# COCOアノテーションファイルのパスを指定
coco_json_path = 'coco/annotations/instances_val2017.json'

# COCO JSONの内容を表示
load_and_display_coco_json(coco_json_path)
