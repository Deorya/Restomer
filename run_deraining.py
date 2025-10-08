import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
import cv2
import os
import numpy as np
from tqdm import tqdm


def get_weights_and_parameters(task, parameters):
    if task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    # ... (可以根据需要添加其他任务)
    else:
        raise ValueError(f"Task {task} not supported.")
    return weights, parameters


def main():
    # 1. 设置任务和输入/输出目录
    task = 'Deraining'
    input_dir = 'input_images'  # 存放你想要处理的图片的文件夹
    output_dir = 'output_images'  # 存放处理结果的文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 2. 加载模型架构和权重
    print("===> Loading model...")
    parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8],
                  'num_refinement_blocks': 4, 'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
                  'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False}
    weights, parameters = get_weights_and_parameters(task, parameters)

    load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)

    # --- 关键修改：自动选择 CPU 或 GPU ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    # -----------------------------------

    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint['params'])
    model.eval()

    # 3. 处理输入目录中的所有图片
    print("===> Processing images...")
    image_files = os.listdir(input_dir)
    for img_file in tqdm(image_files):
        img_path = os.path.join(input_dir, img_file)

        # 读取图片
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).to(device)

        # Restormer 需要输入的图像尺寸是 32 的倍数，进行填充
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
        padh = H - h if h % 32 != 0 else 0
        padw = W - w if w % 32 != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        # 模型推理
        with torch.no_grad():
            restored = model(input_)

        # 移除填充
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        # 保存结果
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_restored.png")
        cv2.imwrite(output_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

    print(f"===> Done! Results saved in '{output_dir}' directory.")


if __name__ == "__main__":
    main()