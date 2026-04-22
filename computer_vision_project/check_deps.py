import torch
import sys

print(f"Python: {sys.version}")
print(f"Torch:  {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError:
    print("OpenCV: NOT INSTALLED")

try:
    import tqdm
    print(f"tqdm:   INSTALLED")
except ImportError:
    print("tqdm:   NOT INSTALLED")

try:
    import pandas
    print(f"pandas: {pandas.__version__}")
except ImportError:
    print("pandas: NOT INSTALLED")
