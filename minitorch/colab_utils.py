import subprocess
import os

def update_package(package_name):
    subprocess.call(['cp', f"/content/{package_name}", f"/content/gdrive/My Drive/PyTorch/MiniTorch/minitorch/{package_name}"])

def copy_minitorch_to_local():
    for fname in os.listdir('/content/gdrive/My Drive/PyTorch/MiniTorch/minitorch'):
        subprocess.call(['cp', f"/content/gdrive/My Drive/PyTorch/MiniTorch/minitorch/{fname}", "/content"])