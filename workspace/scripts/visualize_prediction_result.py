import os
import shutil
import cv2
import numpy as np

def visualize_result(original_img_path, mask_path, save_path, alpha=0.5):
    # 1. 加载原图和掩膜
    img = cv2.imread(original_img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 2. 创建彩色掩膜层 (BGR)
    color_mask = np.zeros_like(img)
    # 类别 1: 叶片 -> 绿色
    color_mask[mask == 1] = [0, 255, 0]
    # 类别 2: 病斑 -> 红色
    color_mask[mask == 2] = [0, 0, 255]

    # 3. 叠加
    overlay = cv2.addWeighted(img, 1, color_mask, alpha, 0)

    # 4. 保存全分辨率结果
    cv2.imwrite(save_path, overlay)

    # 5. 生成一个缩小的预览图（12k 电脑看太卡，缩成 2k）
    h, w = overlay.shape[:2]
    preview = cv2.resize(overlay, (2000

                                       , int(2000 * h / w)))
    cv2.imwrite(save_path.replace(".png", "_preview.jpg"), preview)
    print("可视化完成！")

#visualize_result("/mnt/data/agriculture/blastseg/test/ScreenShot_2026-01-28_133615_410.png", "test_output2/ScreenShot_2026-01-28_133615_410.png", "final_overlay2.png")
val_mask_dir = '/mnt/data/agriculture/nnUNet/workspace/results/Dataset001_LeafBlast/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation'
image_dir = '/mnt/data/agriculture/blastseg/data/dataset_v1_crops/images'

files = os.listdir(val_mask_dir)

for file in files:
    src_path = os.path.join(image_dir, file)

    if os.path.exists(src_path):
        visualize_result(
            src_path,
            os.path.join(val_mask_dir, file),
            os.path.join("/mnt/data/agriculture/nnUNet/workspace/val_output", file)
        )

        # 拆分文件名和扩展名
        name, ext = os.path.splitext(file)
        dst_file = f"{name}_rgb{ext}"
        dst_path = os.path.join(
            "/mnt/data/agriculture/nnUNet/workspace/val_output",
            dst_file
        )

        shutil.copy(src_path, dst_path)
