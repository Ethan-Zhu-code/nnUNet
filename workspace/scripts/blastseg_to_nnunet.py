import os
import shutil
import json
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

def convert_to_nnunet(src_root, output_root, target_dataset_id, task_name):
    # 1. 创建 nnU-Net 目录结构
    nnunet_raw = Path(output_root) / f"Dataset{target_dataset_id:03d}_{task_name}"
    img_tr = nnunet_raw / "imagesTr"
    lab_tr = nnunet_raw / "labelsTr"
    
    img_tr.mkdir(parents=True, exist_ok=True)
    lab_tr.mkdir(parents=True, exist_ok=True)

    src_images = Path(src_root) / "images"
    src_masks = Path(src_root) / "masks"

    if not src_images.exists():
        print(f"Error: Source path {src_images} does not exist!")
        return

    files = [f for f in os.listdir(src_images) if f.endswith('.png')]
    print(f"Found {len(files)} images to process...")
    
    for f in tqdm(files, desc="Splitting Channels"):
        case_id = os.path.splitext(f)[0]
        
        # 读取 RGB 图片
        img_path = str(src_images / f)
        img = cv2.imread(img_path) # 默认读取为 BGR
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # 拆分通道并保存为独立的模态文件
        # nnU-Net 要求：_0000, _0001, _0002
        # img[:, :, 2] is Red, img[:, :, 1] is Green, img[:, :, 0] is Blue (since OpenCV reads as BGR)
        cv2.imwrite(str(img_tr / f"{case_id}_0000.png"), img[:, :, 2]) # Red
        cv2.imwrite(str(img_tr / f"{case_id}_0001.png"), img[:, :, 1]) # Green
        cv2.imwrite(str(img_tr / f"{case_id}_0002.png"), img[:, :, 0]) # Blue
        
        # 复制标签: case_001.png -> case_001.png
        shutil.copy(src_masks / f, lab_tr / f"{case_id}.png")

    # 2. 生成 dataset.json
    dataset_info = {
        "channel_names": {
            "0": "Red",
            "1": "Green",
            "2": "Blue"
        },
        "labels": {
            "background": 0,
            "leaf": 1,
            "lesion": 2
        },
        "numTraining": len(files),
        "file_ending": ".png"
    }
    
    with open(nnunet_raw / "dataset.json", "w") as j:
        json.dump(dataset_info, j, indent=4)

    print(f"Done! Dataset saved at {nnunet_raw}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BlastSeg dataset to nnU-Net format with RGB channel splitting")
    
    parser.add_argument("--src_root", type=str, 
                        default="/mnt/data/agriculture/blastseg/data/dataset_v1_crops",
                        help="Source dataset root directory")
    
    parser.add_argument("--output_root", type=str, 
                        default="nnUNet_raw",
                        help="Output directory where Dataset folder will be created")
                        
    parser.add_argument("--id", type=int, default=1, 
                        help="Dataset ID (integer)")
                        
    parser.add_argument("--task", type=str, default="LeafBlast", 
                        help="Task name")

    args = parser.parse_args()

    convert_to_nnunet(
        src_root=args.src_root,
        output_root=args.output_root,
        target_dataset_id=args.id,
        task_name=args.task
    )
