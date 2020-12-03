import subprocess
from IPython.display import clear_output

def update_package(package_name):
  subprocess.call(['cp', f"/content/{package_name}", f"/content/gdrive/My Drive/PyTorch/utils/{package_name}"])