#check for cuda
# from ultralytics import YOLO
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # Load model on the correct device
# model = YOLO("best.pt").to(device)

# import torch
# print("CUDA Available:", torch.cuda.is_available())
# print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Found")

import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA-compatible GPU found.")
