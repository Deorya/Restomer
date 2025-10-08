import torch

is_available = torch.cuda.is_available()
print(f"Is CUDA available? -> {is_available}")

if is_available:
    print(f"CUDA version detected by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU Index: {torch.cuda.current_device()}")
    print(f"Current GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("!!! CRITICAL: PyTorch cannot find a compatible GPU and CUDA installation.")
    print("!!! This is why the training is running on the CPU.")