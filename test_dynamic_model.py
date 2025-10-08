import torch
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image

# 1. 导入我们修改后的模型架构
#    只要你的 restormer_arch.py 是修改过的版本，这里导入的就是正确的架构
from basicsr.models.archs.restormer_arch import Restormer

# =======================================================================
# ======================== 1. 配置区域 ============================
# =======================================================================

CONFIG = {
    # --- 路径配置 ---
    "input_dir": "input_images",  # 需要去雨的图片文件夹
    "output_dir": "output_images",  # 保存结果的文件夹

    "new_model_weights": "checkpoints_dynamic/model_epoch_100.pth",  # 你训练的新模型权重

    # --- 模型架构参数 ---
    # !! 必须和你训练时的参数完全一致 !!
    "new_model_dim": 48,  # 假设你训练时用的是 64

    # --- 设备配置 ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# 创建输出文件夹
os.makedirs(CONFIG["output_dir"], exist_ok=True)


# =======================================================================
# ======================== 2. 推理主程序 ==========================
# =======================================================================
def main():
    # --- 关键：明确指定 ffn_type='dynamic' ---
    print("Initializing NEW Restormer model (dynamic FFN)...")
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        ffn_type = 'dynamic'  # <--- 明确告诉模型使用动态版本
    ).to(CONFIG["device"])

    # 加载你新训练的权重
    print(f"Loading new weights from: {CONFIG['new_model_weights']}")
    checkpoint = torch.load(CONFIG['new_model_weights'])
    model.load_state_dict(checkpoint)
    model.eval()

    # --- 推理过程 (代码完全不用变) ---
    # ...
    print("New dynamic model processing finished.")


if __name__ == "__main__":
    main()
