#!/bin/bash

# GPU Detection
gpu_info=$(lspci | grep -i 'vga\|3d\|display')

if echo "$gpu_info" | grep -qi "nvidia"; then
    echo "NVIDIA GPU detected"
    GPU_TYPE="nvidia"
elif echo "$gpu_info" | grep -qi "amd\|advanced micro devices\|ati"; then
    echo "AMD GPU detected"
    GPU_TYPE="amd"
else
    echo "No NVIDIA/AMD GPU detected"
    exit 1
fi

# Package Manager Detection
PKG_MGR=""
if command -v pacman &> /dev/null; then
    PKG_MGR="pacman"
elif command -v apt-get &> /dev/null; then
    PKG_MGR="apt"
elif command -v dnf &> /dev/null; then
    PKG_MGR="dnf"
elif command -v zypper &> /dev/null; then
    PKG_MGR="zypper"
elif command -v yum &> /dev/null; then
    PKG_MGR="yum"
fi

if [ -z "$PKG_MGR" ]; then
    echo "No supported package manager found"
    exit 1
fi

echo "Detected package manager: $PKG_MGR"

# GPU Toolkit Installation
case "$PKG_MGR" in
    pacman)
        if [ "$GPU_TYPE" = "nvidia" ]; then
            echo "Installing NVIDIA drivers and CUDA for Arch Linux..."
            sudo pacman -Sy --noconfirm nvidia-dkms cuda
        else  # AMD
            echo "Installing AMD ROCm for Arch Linux..."
            sudo pacman -Sy --noconfirm rocm-opencl-runtime
            echo "For full ROCm stack: sudo pacman -Sy --noconfirm rocm-dkms"
            # Add PyTorch installation
            pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1/
        fi
        ;;
    apt)
        if [ "$GPU_TYPE" = "nvidia" ]; then
            echo "Installing NVIDIA drivers and CUDA for Ubuntu/Debian..."
            sudo apt update
            sudo apt install -y linux-headers-$(uname -r) build-essential
            sudo apt install -y nvidia-driver-535  # Adjust version as needed
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
            sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda
            sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
            echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list
            sudo apt update
            sudo apt install -y cuda
        else  # AMD
            echo "Installing AMD ROCm for Ubuntu..."
            sudo apt update
            sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:rocm/stable
            sudo apt update
            sudo apt install -y rocm-dkms
            # Add PyTorch installation
            pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1/
        fi
        ;;
    dnf)
        if [ "$GPU_TYPE" = "nvidia" ]; then
            echo "Installing NVIDIA drivers and CUDA for Fedora..."
            sudo dnf install -y akmod-nvidia
            sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora38/x86_64/cuda-fedora38.repo
            sudo dnf install -y cuda
        else  # AMD
            echo "Installing AMD ROCm for Fedora..."
            sudo dnf install -y https://repo.radeon.com/amdgpu-install/23.20/23.20-123456/rocm/amdgpu-install-fedora38-23.20-123456.noarch.rpm
            sudo amdgpu-install --usecase=rocm
            # Add PyTorch installation
            pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1/
        fi
        ;;
    zypper)
        if [ "$GPU_TYPE" = "nvidia" ]; then
            echo "Installing NVIDIA drivers and CUDA for openSUSE..."
            sudo zypper addrepo https://developer.download.nvidia.com/compute/cuda/repos/sles15/x86_64/ cuda
            sudo zypper refresh
            sudo zypper install -y cuda
        else  # AMD
            echo "Installing AMD ROCm for openSUSE..."
            sudo zypper addrepo --refresh https://repo.radeon.com/amdgpu-install/23.20/23.20-123456/sles/15.4/x86_64/ amdgpu
            sudo zypper install -y amdgpu-install
            sudo amdgpu-install --usecase=rocm
            # Add PyTorch installation
            pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1/
        fi
        ;;
    yum)
        if [ "$GPU_TYPE" = "nvidia" ]; then
            echo "Installing NVIDIA drivers and CUDA for CentOS/RHEL..."
            sudo yum install -y epel-release
            sudo yum install -y https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-repo-rhel8-12-1-local-12.1.0_530.30-1.x86_64.rpm
            sudo yum clean all
            sudo yum install -y cuda
        else  # AMD
            echo "Installing AMD ROCm for CentOS/RHEL..."
            sudo yum install -y https://repo.radeon.com/amdgpu-install/23.20/23.20-123456/rhel/8.6/x86_64/amdgpu-install-23.20.3-1.el8.noarch.rpm
            sudo amdgpu-install --usecase=rocm
            # Add PyTorch installation
            pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1/
        fi
        ;;
esac

echo "Installation complete! Reboot recommended"
