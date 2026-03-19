import os
import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def visualize_results(mask_dir, raw_img_dir, output_dir, alpha=0.5, copy_raw=False):
    mask_path = Path(mask_dir)
    img_path = Path(raw_img_dir)
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 定义颜色映射 (B, G, R)
    color_map = {
        0: [0, 0, 0],  # 背景: 黑色
        1: [0, 255, 0],  # 叶片: 绿色
        2: [255, 0, 0],  # 类别2: 黄色
        3: [0, 0, 255]  # 类别3: 红色
    }

    mask_files = list(mask_path.glob("*.png"))
    if not mask_files:
        print(f"❌ 错误：在 {mask_dir} 中未找到任何 .png 文件！")
        return

    print(f"开始可视化处理，共 {len(mask_files)} 个样本...")

    for m_file in tqdm(mask_files):
        # 获取基础文件名和后缀，例如 "104_res" 和 ".png"
        base_name = m_file.stem
        suffix = m_file.suffix

        # 1. 读取预测图
        mask = cv2.imread(str(m_file), cv2.IMREAD_GRAYSCALE)
        if mask is None: continue

        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in color_map.items():
            color_mask[mask == label] = color

        # 2. 查找原始图像
        img_f = img_path / m_file.name
        if not img_f.exists():
            img_f = img_path / m_file.with_suffix('.jpg').name

        if img_f.exists():
            raw_img = cv2.imread(str(img_f))

            # 3. 生成叠加图，命名为: 文件名_vis.png
            overlay = cv2.addWeighted(raw_img, 1 - alpha, color_mask, alpha, 0)
            cv2.imwrite(str(save_path / f"{base_name}_vis{suffix}"), overlay)

            # 4. 如果开启了 --copy_raw，原图命名为: 文件名_raw.png
            if copy_raw:
                shutil.copy2(str(img_f), str(save_path / f"{base_name}_raw{suffix}"))
        else:
            # 找不到原图则只保存彩色掩膜，命名为: 文件名_mask.png
            cv2.imwrite(str(save_path / f"{base_name}_mask{suffix}"), color_mask)

    print(f"\n✅ 处理完成！结果保存在: {output_dir}")
    print(f"排序提示：现在文件会以 '{base_name}_raw.png' 和 '{base_name}_vis.png' 的形式成对排列。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nnU-Net 可视化工具（后缀命名版）")
    parser.add_argument("-m", "--mask_dir", type=str, required=True, help="推理结果文件夹")
    parser.add_argument("-i", "--img_dir", type=str, required=True, help="原始彩色图像文件夹")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="保存路径")
    parser.add_argument("-a", "--alpha", type=float, default=0.4, help="透明度")
    parser.add_argument("--copy_raw", action="store_true", help="是否同时拷贝原始彩色图像")

    args = parser.parse_args()
    visualize_results(args.mask_dir, args.img_dir, args.output_dir, args.alpha, args.copy_raw)