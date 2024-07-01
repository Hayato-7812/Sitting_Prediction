import os
import subprocess
from tqdm import tqdm

def download_and_unzip(url, output_dir):
    filename = os.path.basename(url)
    filepath = os.path.join(output_dir, filename)
    
    # ダウンロード
    command = ["wget", url, "-O", filepath]
    subprocess.run(command, check=True)
    
    # 解凍
    command = ["unzip", filepath, "-d", output_dir]
    subprocess.run(command, check=True)
    
    # zipファイルを削除
    os.remove(filepath)

def main():
    base_url = "http://images.cocodataset.org/zips/"
    annotation_url = "http://images.cocodataset.org/annotations/"
    output_dir = "./coco"
    
    # ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # 画像データのダウンロードと解凍
    image_files = ["train2017.zip", "val2017.zip", "test2017.zip", "unlabeled2017.zip"]
    for file in tqdm(image_files, desc="Downloading images"):
        download_and_unzip(base_url + file, image_dir)
    
    # アノテーションデータのダウンロードと解凍
    annotation_files = ["annotations_trainval2017.zip", "stuff_annotations_trainval2017.zip", "image_info_test2017.zip", "image_info_unlabeled2017.zip"]
    for file in tqdm(annotation_files, desc="Downloading annotations"):
        download_and_unzip(annotation_url + file, output_dir)

if __name__ == "__main__":
    main()

