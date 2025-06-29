# 🚀 EasyTrain: Minimalist LLM Training & Inference Toolkit

**EasyTrain** is a lightweight, cross-platform CLI tool that makes training and running large language models (LLMs) as simple as a single command. Designed for developers who value simplicity over complexity.

---

## 🔑 Key Features

- ✅ **Minimalist Design**  
  Just `python3 EasyTrain.py runlm [...]` or `python3 EasyTrain.py trainlm [...]`

- 🧠 **LLM Support**
  - Run: LLaMA, Mistral, Phi, Falcon, etc.
  - Train: LoRA/PEFT support for efficient fine-tuning
  - CUDA acceleration (when available)

- 🌍 **Cross-Platform**
  - Works on Linux, macOS, and Windows (NVIDIA GPUs only)
  - AMD GPU support on Linux only (via ROCm)

- 🧪 **Smart Defaults**
  - Automatic dataset format detection
  - Built-in token streaming
  - PEFT/LoRA support for efficient training

- 🔧 **Modular Architecture**
  - Clean Python modules for extension
  - Importable components:
    ```python
    from text import textgenerationmodel
    from text.train import trainllm
    ```

---

## 🧰 System Requirements

| Component         | Minimum Requirement                  |
|------------------|--------------------------------------|
| OS               | Linux/macOS/Windows (Linux preferred) |
| Python           | 3.8+                                |
| GPU (Optional)   | CUDA-capable NVIDIA GPU (for speed), or AMD ROCm-capable GPU |
| Disk Space       | 20GB+ for models (varies by model)   |

**Note:** AMD ROCm support only available on Linux.

---

## 🛠️ Installation

### 🐧 Linux / macOS

```bash
# Make installer executable
chmod +x autoinstall.sh

# Run installer
./autoinstall.sh
```

🪟 Windows

**Currently no AMD GPU support. For NVIDIA GPUs:**

```bash

# Install CUDA dependencies manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install EasyTrain dependencies
pip install transformers datasets accelerate peft

```

🚀 Quick Start
**Run a Model**

```bash

python3 EasyTrain.py runlm \
  --model meta-llama/Llama-2-7b-hf \
  --prompt "Explain quantum computing in simple terms" \
  --stream

```

**Train with LoRA**

```bash

python3 EasyTrain.py trainlm \
  --model meta-llama/Llama-2-7b-hf \
  --dataset training_data.jsonl \
  --use-peft \
  --lora-rank 64 \
  --lora-alpha 128 \
  --output-dir ./fine_tuned_model

```

**📦 CLI Arguments**
runlm Mode (Inference)
Argument 	Default 	Description
--model 	REQUIRED 	Model name/path
--prompt 	REQUIRED 	Input text
--system-prompt 	"You are an AI assistant" 	Context prompt
--temperature 	0.7 	Sampling temperature
--stream 	False 	Stream tokens in real-time
trainlm Mode (Training)
Argument 	Default 	Description
--model 	REQUIRED 	Base model name/path
--dataset 	REQUIRED 	Dataset path (JSON/JSONL/CSV/TXT)
--use-peft 	False 	Enable PEFT/LoRA training
--lora-rank 	8 	LoRA attention dimension
--lora-alpha 	16 	LoRA scaling parameter
--training-steps 	400 	Total training steps (min 400)
📁 Project Structure

EasyTrain/
├── EasyTrain.py          # Main CLI interface
├── text/
│   ├── textgenerationmodel.py  # Inference model
│   └── train/
│       └── trainllm.py         # Training module
├── autoinstall.sh        # Environment setup script
└── README.md             # This file

📚 Credits

Built with 🧠 HuggingFace Transformers
Optimized with 🔧 PEFT (LoRA)
Streamed with 📡 TextStreamer

📜 License

MIT License - See LICENSE for details
🔭 Contributing

Contributions are welcome! Whether you want to:

Add Windows AMD support when available
Improve dataset format detection
Add new training features
Optimize inference speed

**COMMON ISSUES**

1) When running autoinstall.sh, the script might sometimes falsely detect a gpu (at least on my laptop) which I have not found a way around yet but here is the fix, if you run cpu only:
For apt:

```bash
sudo apt install python3 python3-pip
pip3 install -r requirements.txt && pip3 install torch (--break-system-requirements flag if needed)
```
dnf:

```bash
sudo dnf install python3 python3-pip
pip3 install -r requirements.txt && pip3 install torch (--break-system-requirements flag if needed)
```
yum:

```bash
sudo yum install python3 python3-pip
pip3 install -r requirements.txt && pip3 install torch (--break-system-requirements flag if needed)
```
pacman:

```bash
sudo pacman -Syu install python3 python3-pip
pip3 install -r requirements.txt && pip3 install torch (--break-system-requirements flag if needed)
```

MacOS:

```bash
brew install python
pip3 install -r requirements.txt && pip3 install torch (--break-system-requirements flag if needed)
```
2) Sometimes AMD ROCm fails to install due to bad requests and stuff. It may also happen if you are using a lesser known distro like Oracle Linux or Azure Linux. To install fresh ROCm:

Ubuntu 24.04:

```bash
wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/noble/amdgpu-install_6.4.60401-1_all.deb
sudo apt install ./amdgpu-install_6.4.60401-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
```
Ubuntu 22.04:
```bash
wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/jammy/amdgpu-install_6.4.60401-1_all.deb
sudo apt install ./amdgpu-install_6.4.60401-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
```
Debian 12:
```bash
wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/jammy/amdgpu-install_6.4.60401-1_all.deb
sudo apt install ./amdgpu-install_6.4.60401-1_all.deb
sudo apt update
sudo apt install -y python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
```
Red Hat 9.6:
```bash
sudo dnf install https://repo.radeon.com/amdgpu-install/6.4.1/rhel/9.6/amdgpu-install-6.4.60401-1.el9.noarch.rpm
sudo dnf clean all
wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
sudo rpm -ivh epel-release-latest-9.noarch.rpm
sudo dnf install dnf-plugin-config-manager
sudo crb enable
sudo dnf install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo dnf install rocm
```
Red Hat 9.5:
```bash
sudo dnf install https://repo.radeon.com/amdgpu-install/6.4.1/rhel/9.5/amdgpu-install-6.4.60401-1.el9.noarch.rpm
sudo dnf clean all
sudo dnf install "kernel-headers-$(uname -r)" "kernel-devel-$(uname -r)" "kernel-devel-matched-$(uname -r)"
sudo dnf install amdgpu-dkms
```
Red Hat 9.4:
```bash
sudo dnf install https://repo.radeon.com/amdgpu-install/6.4.1/rhel/9.4/amdgpu-install-6.4.60401-1.el9.noarch.rpm
sudo dnf clean all
sudo dnf install "kernel-headers-$(uname -r)" "kernel-devel-$(uname -r)" "kernel-devel-matched-$(uname -r)"
sudo dnf install amdgpu-dkms
```
Red Hat 8.10
```bash
sudo dnf install https://repo.radeon.com/amdgpu-install/6.4.1/rhel/8.10/amdgpu-install-6.4.60401-1.el8.noarch.rpm
sudo dnf clean all
sudo dnf install "kernel-headers-$(uname -r)" "kernel-devel-$(uname -r)"
sudo dnf install amdgpu-dkms
```
Oracle Linux
```bash
sudo dnf install https://repo.radeon.com/amdgpu-install/6.4.1/el/9.5/amdgpu-install-6.4.60401-1.el9.noarch.rpm
sudo dnf clean all
sudo dnf install "kernel-uek-devel-$(uname -r)"
sudo dnf install amdgpu-dkms
```
SUSE
```bash
sudo SUSEConnect -p sle-module-desktop-applications/15.6/x86_64
sudo SUSEConnect -p sle-module-development-tools/15.6/x86_64
sudo SUSEConnect -p PackageHub/15.6/x86_64
sudo zypper install zypper
sudo zypper --no-gpg-checks install https://repo.radeon.com/amdgpu-install/6.4.1/sle/15.6/amdgpu-install-6.4.60401-1.noarch.rpm
sudo zypper --gpg-auto-import-keys refresh
sudo zypper install kernel-default-devel
sudo zypper install amdgpu-dkms
```
Azure Linux
```bash
sudo tdnf install "kernel-headers-$(uname -r)" "kernel-devel-$(uname -r)"
sudo tdnf install azurelinux-repos-amd
sudo tdnf repolist --refresh
sudo tdnf install amdgpu
```

Then install PyTorch for ROCm:
```bash
 pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/
```

Just fork and submit a PR!
📬 Feedback

Have ideas to make this simpler? Found a bug?
Open an issue or PR on GitHub!

