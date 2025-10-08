import torch
import os
import glob
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm
import torch.nn.functional as F

# 导入我们升级后的、灵活的 Restormer 架构
from basicsr.models.archs.restormer_arch import Restormer

# --- 配置 ---
CONFIG = {
    "input_dir": "input_images",
    "output_dir": "results_original_model",
    "original_weights": "Deraining/pretrained_models/deraining.pth",  # 原始模型权重
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def main():
    # --- 关键：以 'static' 模式创建模型 ---
    # 我们不传递 ffn_type，所以它会使用默认值 'static'
    print("Initializing ORIGINAL Restormer model (static FFN)...")
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    ).to(CONFIG["device"])

    # 加载原始权重
    print(f"Loading original weights from: {CONFIG['original_weights']}")
    checkpoint = torch.load(CONFIG['original_weights'])
    # 原始权重的key是 'params'
    model.load_state_dict(checkpoint['params'])
    model.eval()

    # --- 推理过程 (与之前的脚本相同) ---
    input_paths = sorted(glob.glob(os.path.join(CONFIG["input_dir"], '*')))
    with torch.no_grad():
        for img_path in tqdm(input_paths, desc="Processing with original model"):
            # ... (推理和保存的代码) ...
            input_img = Image.open(img_path).convert('RGB')
            input_tensor = to_tensor(input_img).unsqueeze(0).to(CONFIG["device"])
            # 1. 检查尺寸并计算需要的 padding
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = input_tensor.shape
            divisor = 8 # Restormer 需要是 8 的倍数
            if h % divisor != 0:
                mod_pad_h = divisor - h % divisor
            if w % divisor != 0:
                mod_pad_w = divisor - w % divisor
            
            # 2. 对输入进行 padding
            # F.pad 的参数是 (左, 右, 上, 下)
            input_tensor = F.pad(input_tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            restored_tensor = model(input_tensor).clamp(0, 1)
            restored_img = to_pil_image(restored_tensor.squeeze(0).cpu())
            restored_img.save(os.path.join(CONFIG["output_dir"], os.path.basename(img_path)))
    print("Original model processing finished.")


if __name__ == "__main__":
    main()
