import sys
import torch
import torchvision
import flask
import flask_cors
import sklearn
import pandas
import numpy
from PIL import Image

print("=== Python & Package Environment Check ===\n")

print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"Flask Version: {flask.__version__}")
print(f"Flask-Cors Version: {flask_cors.__version__}")
print(f"Scikit-Learn Version: {sklearn.__version__}")
print(f"Pandas Version: {pandas.__version__}")
print(f"NumPy Version: {numpy.__version__}")
print(f"Pillow Version: {Image.__version__}")