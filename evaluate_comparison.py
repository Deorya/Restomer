import torch
import torch.nn.functional as F
import os
import glob
import time
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor

# 质量评估库
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import cv2


# 导入我们升级后的、灵活的 Restormer 架构
from basicsr.models.archs.restormer_arch import Restormer

# =======================================================================
# ======================== 1. 配置区域 ============================
# =======================================================================

CONFIG = {
    # --- 路径配置 ---
    "test_input_dir": "/path/to/your/test/images/input",
    "test_target_dir": "/path/to/your/test/images/target",

    "original_model_weights": "Deraining/pretrained_models/deraining.pth",
    "new_model_weights": "checkpoints_dynamic/model_epoch_100.pth",  # 确保这个文件是完好的

    # --- 模型架构参数 ---
    # !! 确保这些参数与模型文件完全匹配 !!
    "original_model_params": {
        "dim": 48,
        "ffn_type": "static"  # 使用默认的静态FFN
    },
    "new_model_params": {
        "dim": 64,  # 假设你改为了64
        "ffn_type": "dynamic"  # 使用我们的动态FFN
    },

    # --- 评估配置 ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =======================================================================
# ======================== 2. 评估主程序 ==========================
# =======================================================================

def load_models():
    """根据配置加载原始模型和新模型"""
    models = {}
    print("Loading models...")

    # --- 加载原始模型 ---
    print("  - Initializing Original Model...")
    original_params = CONFIG["original_model_params"]
    original_model = Restormer(
        dim=original_params["dim"],
        ffn_type=original_params["ffn_type"]
        # 你可以添加其他非默认参数
    ).to(CONFIG["device"])

    original_checkpoint = torch.load(CONFIG["original_model_weights"])
    original_model.load_state_dict(original_checkpoint.get('params', original_checkpoint))
    original_model.eval()
    models["original"] = original_model

    # --- 加载新模型 ---
    print("  - Initializing New Dynamic Model...")
    new_params = CONFIG["new_model_params"]
    new_model = Restormer(
        dim=new_params["dim"],
        ffn_type=new_params["ffn_type"]
    ).to(CONFIG["device"])

    new_checkpoint = torch.load(CONFIG["new_model_weights"])
    new_model.load_state_dict(new_checkpoint)
    new_model.eval()
    models["new"] = new_model

    print("Models loaded successfully.")
    return models


def evaluate():
    models = load_models()

    lpips_evaluator = lpips.LPIPS(net='alex').to(CONFIG["device"])

    input_paths = sorted(glob.glob(os.path.join(CONFIG["test_input_dir"], '*')))
    target_paths = sorted(glob.glob(os.path.join(CONFIG["test_target_dir"], '*')))

    results = {
        'original': {'psnr': [], 'ssim': [], 'lpips': [], 'time': []},
        'new': {'psnr': [], 'ssim': [], 'lpips': [], 'time': []}
    }

    # --- 循环处理每张图片 ---
    with torch.no_grad():
        for i in tqdm(range(len(input_paths)), desc="Evaluating Images"):
            input_path = input_paths[i]
            target_path = target_paths[i]

            input_img_pil = Image.open(input_path).convert('RGB')
            input_tensor = to_tensor(input_img_pil).unsqueeze(0).to(CONFIG["device"])

            # 准备目标图像
            target_np = np.array(Image.open(target_path).convert('RGB'))
            target_tensor_lpips = lpips.im2tensor(lpips.load_image(target_path)).to(CONFIG["device"])

            # --- 自动 Padding ---
            h_old, w_old = input_tensor.shape[2], input_tensor.shape[3]
            divisor = 8
            h_pad = (divisor - h_old % divisor) % divisor
            w_pad = (divisor - w_old % divisor) % divisor
            input_tensor = F.pad(input_tensor, (0, w_pad, 0, h_pad), 'reflect')

            # --- 对两个模型进行评估 ---
            for model_name, model in models.items():

                # 推理和计时
                if CONFIG["device"] == "cuda": torch.cuda.synchronize()
                start_time = time.time()
                output_tensor = model(input_tensor).clamp(0, 1)
                if CONFIG["device"] == "cuda": torch.cuda.synchronize()
                results[model_name]['time'].append(time.time() - start_time)

                # 自动 Cropping
                output_tensor = output_tensor[:, :, :h_old, :w_old]

                # 转换输出格式以进行评估
                output_np = (output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

                # 计算质量指标
                results[model_name]['psnr'].append(psnr(target_np, output_np, data_range=255))
                results[model_name]['ssim'].append(
                    ssim(target_np, output_np, multichannel=True, data_range=255, channel_axis=2))
                results[model_name]['lpips'].append(lpips_evaluator(target_tensor_lpips, output_tensor * 2 - 1).item())
                output_cv2_gray = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)


    # --- 计算效率指标 ---
    efficiency_results = {'original': {}, 'new': {}}
    for model_name, model in models.items():
        weights_path = CONFIG[f"{model_name}_model_weights"]
        efficiency_results[model_name]['size_mb'] = os.path.getsize(weights_path) / (1024 * 1024)
        efficiency_results[model_name]['params_m'] = sum(p.numel() for p in model.parameters()) / 1e6
        avg_time = np.mean(results[model_name]['time'])
        efficiency_results[model_name]['inference_time_ms'] = avg_time * 1000

        if CONFIG["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
            # 在一个典型的输入上运行一次来记录峰值内存
            model(input_tensor)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            efficiency_results[model_name]['vram_mb'] = peak_memory

    print_report(results, efficiency_results)


def print_report(results, efficiency_results):
    print("\n\n" + "=" * 80)
    print(" " * 28 + "MODEL PERFORMANCE REPORT")
    print("=" * 80)

    print(f"{'Metric':<35} | {'Original Model':<20} | {'New Dynamic Model':<20}")
    print("-" * 80)

    # 打印质量指标
    print("--- Image Quality Metrics (Average on Test Set) ---")
    for metric in ['psnr', 'ssim', 'lpips']:
        original_avg = np.mean(results['original'][metric])
        new_avg = np.mean(results['new'][metric])

        is_better = False
        if metric in ['psnr', 'ssim']:
            is_better = new_avg > original_avg
            metric_display = f"{metric.upper()} (Higher is better)"
        else:  # lpips
            is_better = new_avg < original_avg
            metric_display = f"{metric.upper()} (Lower is better)"

        arrow = "↑ (Better)" if is_better else "↓ (Worse)"

        print(f"{metric_display:<35} | {original_avg:<20.4f} | {new_avg:<20.4f} {arrow}")

    # 打印效率指标
    print("\n--- Efficiency & Portability Metrics ---")
    for metric_name, unit, direction in [('size_mb', 'MB', 'Lower'), ('params_m', 'M', 'Lower'),
                                         ('inference_time_ms', 'ms/img', 'Lower'),
                                         ('vram_mb', 'MB', 'Lower')]:
        if metric_name not in efficiency_results['original']: continue

        original_val = efficiency_results['original'][metric_name]
        new_val = efficiency_results['new'][metric_name]

        is_better = (new_val < original_val) if direction == 'Lower' else (new_val > original_val)
        arrow = "↓ (Better)" if is_better else "↑ (Worse)"

        print(
            f"{metric_name.replace('_', ' ').title() + f' ({unit})':<35} | {original_val:<20.2f} | {new_val:<20.2f} {arrow}")

    print("=" * 80)


if __name__ == "__main__":
    evaluate()