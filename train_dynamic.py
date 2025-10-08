import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm

# ----------------- 1. 导入我们修改后的模型 -----------------
# 确保这个文件可以找到 basicsr.models.archs.restormer_arch
# 你可能需要将项目根目录添加到 PYTHONPATH
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basicsr.models.archs.restormer_arch import Restormer

# ----------------- 2. 配置参数 -----------------
CONFIG = {
    "train_input_dir": "Deraining/Datasets/train/Rain100L/input",
    "train_target_dir": "Deraining/Datasets/train/Rain100L/target",
    "pretrained_weights_path": "Deraining/pretrained_models/deraining.pth",  # 确保预训练权重文件在项目根目录
    "checkpoint_save_dir": "./checkpoints_dynamic",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # --- 针对 GTX 1050 (4GB) 的推荐配置 ---
    "image_size": 128,
    "batch_size": 2,  # 物理批次大小
    "gradient_accumulation_steps": 4,  # 梯度累积步数 (有效批次大小 = 2 * 4 = 8)

    "learning_rate": 1e-4,
    "num_epochs": 100,
}
# 创建保存模型的文件夹
os.makedirs(CONFIG["checkpoint_save_dir"], exist_ok=True)


# ----------------- 3. 自定义数据集类 -----------------
class RestorationDataset(Dataset):
    def __init__(self, input_dir, target_dir, image_size):
        self.input_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
        self.target_paths = sorted(glob.glob(os.path.join(target_dir, '*.png')))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # 你可以添加更多的 normalization
        ])
        assert len(self.input_paths) == len(
            self.target_paths), "Input and Target directories must have the same number of images."

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_paths[idx]).convert('RGB')
        target_image = Image.open(self.target_paths[idx]).convert('RGB')
        return self.transform(input_image), self.transform(target_image)


# ----------------- 4. 核心：部分加载权重的函数 -----------------
def load_partial_weights(model, pretrained_path):
    print("=" * 50)
    print(f"Loading partial weights from: {pretrained_path}")

    # 加载预训练权重字典
    pretrained_dict = torch.load(pretrained_path)
    # 原始仓库的权重通常保存在 'params' key 下
    if 'params' in pretrained_dict:
        pretrained_dict = pretrained_dict['params']

    # 获取当前模型的 state_dict
    model_dict = model.state_dict()

    # 1. 筛选出不需要加载的权重 (所有与 ffn 相关的)
    #    并创建要加载的新字典
    weights_to_load = {}
    skipped_weights = []
    loaded_weights = []

    for k, v in pretrained_dict.items():
        # 核心判断逻辑：如果key中不包含 'ffn'，并且在当前模型中存在，则加载
        if 'ffn' not in k and k in model_dict:
            # 还可以添加形状匹配检查以增加鲁棒性
            if v.shape == model_dict[k].shape:
                weights_to_load[k] = v
                loaded_weights.append(k)
            else:
                print(f"  - Shape mismatch, skipping: {k}")
                skipped_weights.append(k)
        else:
            skipped_weights.append(k)

    # 2. 更新当前模型的 state_dict
    model_dict.update(weights_to_load)

    # 3. 将更新后的 state_dict 加载回模型
    model.load_state_dict(model_dict)

    print(f"\nSuccessfully loaded {len(loaded_weights)} layers.")
    print(f"Skipped {len(skipped_weights)} layers (mostly ffn layers).")
    if len(skipped_weights) < 20:  # 打印少量被跳过的层以供检查
        print("A few skipped keys:", skipped_weights[:5])
    print("=" * 50)

    return model


# ----------------- 5. 主训练流程 -----------------
def main():
    # 初始化模型
    # Restormer的参数，请根据你使用的预训练模型进行调整
    # 例如，去雨模型通常 dim=48, num_blocks=[4,6,6,8], num_refinement_blocks=4
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
    )
    model.to(CONFIG["device"])

    # 加载部分权重
    model = load_partial_weights(model, CONFIG["pretrained_weights_path"])

    # 准备数据
    train_dataset = RestorationDataset(CONFIG["train_input_dir"], CONFIG["train_target_dir"], CONFIG["image_size"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4,
                              pin_memory=True)

    # 定义损失函数和优化器
    criterion = nn.L1Loss()  # L1 Loss (MAE) 在图像恢复中很常用
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    # 学习率调度器（可选，但推荐）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)

    # 开始训练
    print("\nStarting training with DynamicFeedForward...")
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        for inputs, targets in progress_bar:
            inputs = inputs.to(CONFIG["device"])
            targets = targets.to(CONFIG["device"])

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1} Summary: Average Loss: {avg_epoch_loss:.6f}, Current LR: {scheduler.get_last_lr()[0]:.6f}")

        # 更新学习率
        scheduler.step()

        # 保存模型
        if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次
            save_path = os.path.join(CONFIG["checkpoint_save_dir"], f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

    print("\nTraining finished!")


if __name__ == "__main__":
    main()