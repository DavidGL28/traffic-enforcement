import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print(
    "Current GPU:",
    torch.cuda.current_device() if torch.cuda.is_available() else "No GPU",
)
