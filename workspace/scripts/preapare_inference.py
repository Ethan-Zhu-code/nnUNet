import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def prepare_inference_data(src_dir, save_dir):
    """
    将普通 RGB 图像转化为 nnU-Net 推理所需的 3 通道独立文件格式
    """
    src_path = Path(src_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    img_exts = ['*.png', '*.jpg', '*.jpeg', '*.tif']
    files = []
    for ext in img_exts:
        files.extend(list(src_path.glob(ext)))

    if not files:
        print(f"错误: 在 {src_dir} 中没找到图像文件！")
        return

    print(f"找到 {len(files)} 张图像，准备进行格式转化...")

    for f_path in tqdm(files):
        case_id = f_path.stem  # 获取文件名（不含后缀）

        # 读取图像
        img = cv2.imread(str(f_path))
        if img is None:
            print(f"跳过无法读取的文件: {f_path}")
            continue

        # 如果图像不是 3 通道的，强制转为 3 通道（处理偶尔出现的灰度图）
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # nnU-Net 训练时如果你用的是 Red(0000), Green(0001), Blue(0002)
        # OpenCV 默认读取顺序是 BGR，所以对应关系如下：
        cv2.imwrite(str(save_path / f"{case_id}_0000.png"), img[:, :, 2])  # Red
        cv2.imwrite(str(save_path / f"{case_id}_0001.png"), img[:, :, 1])  # Green
        cv2.imwrite(str(save_path / f"{case_id}_0002.png"), img[:, :, 0])  # Blue

    print(f"\n转化完成！推理输入数据已保存至: {save_dir}")
    print("你可以现在运行推理命令了，例如:")
    print(f"nnUNetv2_predict -i {save_dir} -o ./inference_results -d 1 -c 2d -f 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备 nnU-Net 推理输入数据")
    parser.add_argument("-i", "--input", type=str, required=True, help="原始 RGB 图像文件夹路径")
    parser.add_argument("-os", "--output", type=str, required=True, help="nnU-Net 格式图像输出路径")

    args = parser.parse_args()
    prepare_inference_data(args.input, args.output)