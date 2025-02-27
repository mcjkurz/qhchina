import subprocess
import sys

def install_package(package_name):
    package_name = package_name.lower().strip()
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
