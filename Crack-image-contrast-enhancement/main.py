import cv2
import matplotlib.pyplot as plt
import os
from linear_processing import linear_hist_equalization
from tanh_processing import tanh_hist_equalization
from contrast_evaluation import evaluate_contrast  # 如果有这个模块


def process_and_save_image(img_path, output_dir, method='linear'):
    print(f"Processing {img_path} using {method}...")

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image {img_path}")
        return

    # 根据选择的方法处理图像
    if method == 'linear':
        gray, result, orig_hist, equalized_hist = linear_hist_equalization(img)
    elif method == 'tanh':
        gray, result, orig_hist, equalized_hist = tanh_hist_equalization(img)
    else:
        print(f"Unsupported method: {method}")
        return

    # 计算对比度
    orig_contrast = evaluate_contrast(gray)
    enhanced_contrast = evaluate_contrast(result)

    # 保存结果图像
    result_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(result_path, result)
    print(f"Processed image saved to {result_path}")

    # 显示结果图片和直方图
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.title("Original Image")
    plt.imshow(gray, cmap='gray')

    plt.subplot(3, 2, 2)
    plt.title("Image after processing")
    plt.imshow(result, cmap='gray')

    plt.subplot(3, 2, 3)
    plt.title("Original Histogram")
    plt.plot(orig_hist, color='blue')

    plt.subplot(3, 2, 4)
    plt.title("Histogram after processing")
    plt.plot(equalized_hist, color='green')

    plt.subplot(3, 2, 5)
    plt.title(f"Original Contrast: {orig_contrast:.2f}")
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.title(f"Enhanced Contrast: {enhanced_contrast:.2f}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_dir = 'D:/Program File/Crack-image-contrast-enhancement/images/Original lmage'  # 替换为实际图像文件夹路径
    output_base_dir = 'D:/Program File/Crack-image-contrast-enhancement/images/lmage after processing/output'  # 替换为实际输出文件夹路径

    # 检查图像目录是否存在
    if not os.path.exists(img_dir):
        #print(f"Image directory {img_dir} does not exist.")
        exit(1)

    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if not img_paths:
        #print(f"No images found in directory {img_dir}.")
        exit(1)

    # 提示用户选择算法
    method = input("请选择要使用的算法 (linear/tanh): ").strip().lower()
    if method not in ['linear', 'tanh']:
        raise ValueError("Unsupported method: {}".format(method))

    # 创建对应算法的输出目录
    output_dir = os.path.join(output_base_dir, method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理所有图像
    for img_path in img_paths:
        process_and_save_image(img_path, output_dir, method=method)

    print(f"图像处理完成，结果保存在 {output_dir}")
