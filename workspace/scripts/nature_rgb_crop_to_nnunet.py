import cv2
import os


def prepare_test_image(img_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # 拆分通道并保存
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_0000.png"), img[:, :, 2])  # R
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_0001.png"), img[:, :, 1])  # G
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_0002.png"), img[:, :, 0])  # B
    print(f"准备就绪：{output_dir}")


prepare_test_image("/mnt/data/agriculture/blastseg/test/ScreenShot_2026-01-28_133615_410.png", "test_input")