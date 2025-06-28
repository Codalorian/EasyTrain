import os
import platform

if platform.system() == "Linux":
    os.system("chmod +x autoinstall.sh && ./autoinstall.sh")
if platform.system() == "Windows":
    os.system("autoinstall.bat")