import torch
import torch.nn as nn
import torchvision.models as models
import torchxrayvision as xrv
import os

BASE_DIR     = r"C:\Ekanath\College\Sem6\xray-search-system"
CHEXNET_PATH = os.path.join(BASE_DIR, "chexnet.pth")

print("Downloading NIH chest X-ray model via torchxrayvision...")

# Download DenseNet trained on NIH ChestX-ray14
# This is the same dataset you are using
model = xrv.models.DenseNet(weights="densenet121-res224-nih")

print("Downloaded successfully!")
print("Saving weights to chexnet.pth ...")

torch.save(
    {"state_dict": model.state_dict(), "source": "torchxrayvision-nih"},
    CHEXNET_PATH
)

print(f"Saved to: {CHEXNET_PATH}")
print("\nNow run: python model.py")