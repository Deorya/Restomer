import torch
import os
import glob
import time
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# 质量评估库
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import cv2
from piq import niqe
# 导入模型架构
from basicsr.models.archs.restormer_arch import Restormer

# =======================================================================
# ======================== 1. 配置区域 ============================
# =======================================================================

CONFIG = {
    # --- 路径配置 ---
    "test_input_dir": "Deraining/Datasets/test/input",
    "test_target_dir": "Deraining/Datasets/test/target",

    "original_model_weights": "Deraining/pretrained_models/deraining.pth",  # 原始Restormer权重
    "new_model_weights": "checkpoints_dynamic/model_epoch_100.pth",  # 你训练的新模型权重

    # --- 模型架构参数 ---
    # !! 确保这些参数与模型训练时完全一致 !!
    "original_model_dim": 48,
    "new_model_dim": 48,  # 假设你改为了64

    # --- 评估配置 ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =======================================================================
# ======================== 2. 评估主程序 ==========================
# =======================================================================

def load_models():
    """加载原始模型和我们修改后的新模型"""
    print("Loading models...")
    # 加载原始模型
    original_model = Restormer(dim=CONFIG["original_model_dim"])
    original_checkpoint = torch.load(CONFIG["original_model_weights"])
    original_model.load_state_dict(
        original_checkpoint['params'] if 'params' in original_checkpoint else original_checkpoint)
    original_model.to(CONFIG["device"])
    original_model.eval()

    # 加载新模型 (它内部使用了 DynamicFeedForward)
    new_model = Restormer(dim=CONFIG["new_model_dim"])  # 使用新的维度
    new_checkpoint = torch.load(CONFIG["new_model_weights"])
    new_model.load_state_dict(new_checkpoint)
    new_model.to(CONFIG["device"])
    new_model.eval()

    print("Models loaded successfully.")
    return {"original": original_model, "new": new_model}


def evaluate():
    models = load_models()

    # 初始化LPIPS评估器
    lpips_evaluator = lpips.LPIPS(net='alex').to(CONFIG["device"])

    # 获取测试图片列表
    input_paths = sorted(glob.glob(os.path.join(CONFIG["test_input_dir"], '*.png')))
    target_paths = sorted(glob.glob(os.path.join(CONFIG["test_target_dir"], '*.png')))

    # 存储所有指标结果的字典
    results = {
        'original': {'psnr': [], 'ssim': [], 'lpips': [], 'niqe': [], 'time': []},
        'new': {'psnr': [], 'ssim': [], 'lpips': [], 'niqe': [], 'time': []}
    }

    transform = transforms.ToTensor()

    with torch.no_grad():
        for i in tqdm(range(len(input_paths)), desc="Evaluating Images"):
            input_path = input_paths[i]
            target_path = target_paths[i]

            input_img_pil = Image.open(input_path).convert('RGB')
            target_img_pil = Image.open(target_path).convert('RGB')

            # --- 准备数据 ---
            input_tensor = transform(input_img_pil).unsqueeze(0).to(CONFIG["device"])

            # 将目标图像转换为 numpy (H, W, C) for PSNR/SSIM
            target_np = np.array(target_img_pil)
            # 将目标图像转换为 tensor for LPIPS
            target_tensor_lpips = lpips.im2tensor(lpips.load_image(target_path)).to(CONFIG["device"])

            # --- 对两个模型进行评估 ---
            for model_name, model in models.items():

                # -- 运行时间 --
                if CONFIG["device"] == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()

                # 模型推理
                output_tensor = model(input_tensor).clamp(0, 1)

                if CONFIG["device"] == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()
                results[model_name]['time'].append(end_time - start_time)

                # --- 计算质量指标 ---
                output_np = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
                output_np = (output_np * 255.0).round().astype(np.uint8)

                # PSNR & SSIM
                results[model_name]['psnr'].append(psnr(target_np, output_np, data_range=255))
                results[model_name]['ssim'].append(ssim(target_np, output_np, multichannel=True, data_range=255))

                # LPIPS
                output_tensor_lpips = output_tensor * 2 - 1  # LPIPS 需要 [-1, 1] 范围
                results[model_name]['lpips'].append(lpips_evaluator(target_tensor_lpips, output_tensor_lpips).item())

                # NIQE (在CPU上用OpenCV格式计算)
                output_cv2 = cv2.cvtColor(output_np, cv2.COLOR_RGB_GRAY)
                results[model_name]['niqe'].append(niqe.score(output_cv2))

    # --- 计算效率指标 ---
    efficiency_results = {'original': {}, 'new': {}}
    for model_name, model in models.items():
        # 模型大小 (MB)
        weights_path = CONFIG[f"{model_name}_model_weights"]
        efficiency_results[model_name]['size_mb'] = os.path.getsize(weights_path) / (1024 * 1024)

        # 参数量 (M)
        efficiency_results[model_name]['params_m'] = sum(p.numel() for p in model.parameters()) / 1e6

        # 推理时间 (ms/image) & FPS
        avg_time = np.mean(results[model_name]['time'])
        efficiency_results[model_name]['inference_time_ms'] = avg_time * 1000
        efficiency_results[model_name]['fps'] = 1 / avg_time

        # 内存占用 (MB) - 需要GPU
        if CONFIG["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
            # 运行一次推理来记录峰值内存
            model(input_tensor)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            efficiency_results[model_name]['vram_mb'] = peak_memory

    # --- 打印最终的对比报告 ---
    print_report(results, efficiency_results)


def print_report(results, efficiency_results):
    print("\n\n" + "=" * 80)
    print(" " * 28 + "MODEL PERFORMANCE REPORT")
    print("=" * 80)

    print(f"{'Metric':<25} | {'Original Model':<25} | {'New Dynamic Model':<25}")
    print("-" * 80)

    # 打印质量指标
    print("--- Image Quality Metrics (Average) ---")
    for metric in ['psnr', 'ssim', 'lpips', 'niqe']:
        original_avg = np.mean(results['original'][metric])
        new_avg = np.mean(results['new'][metric])
        arrow_psnr_ssim = "↑" if new_avg > original_avg else "↓"
        arrow_lpips_niqe = "↓" if new_avg < original_avg else "↑"
        arrow = arrow_psnr_ssim if metric in ['psnr', 'ssim'] else arrow_lpips_niqe

        print(
            f"{metric.upper() + ' (Higher is better)' if metric in ['psnr', 'ssim'] else metric.upper() + ' (Lower is better)':<25} | {original_avg:<25.4f} | {new_avg:<25.4f} ({arrow})")

    # 打印效率指标
    print("\n--- Efficiency & Portability Metrics ---")
    for metric_name, unit, direction in [('size_mb', 'MB', 'Lower'), ('params_m', 'M', 'Lower'),
                                         ('inference_time_ms', 'ms/img', 'Lower'), ('fps', 'FPS', 'Higher'),
                                         ('vram_mb', 'MB', 'Lower')]:
        if metric_name not in efficiency_results['original']: continue

        original_val = efficiency_results['original'][metric_name]
        new_val = efficiency_results['new'][metric_name]

        if direction == 'Higher':
            arrow = "↑" if new_val > original_val else "↓"
        else:  # Lower is better
            arrow = "↓" if new_val < original_val else "↑"

        print(
            f"{metric_name.replace('_', ' ').title() + f' ({unit}, {direction} is better)':<25} | {original_val:<25.2f} | {new_val:<25.2f} ({arrow})")

    print("=" * 80)


if __name__ == "__main__":
    evaluate()