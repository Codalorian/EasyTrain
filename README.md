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
| GPU (Optional)   | CUDA-capable NVIDIA GPU (for speed) |
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

🪟 Windows

    Currently no AMD GPU support. For NVIDIA GPUs:

# Install CUDA dependencies manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install EasyTrain dependencies
pip install transformers datasets accelerate peft

🚀 Quick Start
Run a Model

python3 EasyTrain.py runlm \
  --model meta-llama/Llama-2-7b-hf \
  --prompt "Explain quantum computing in simple terms" \
  --stream

Train with LoRA

python3 EasyTrain.py trainlm \
  --model meta-llama/Llama-2-7b-hf \
  --dataset training_data.jsonl \
  --use-peft \
  --lora-rank 64 \
  --lora-alpha 128 \
  --output-dir ./fine_tuned_model

📦 CLI Arguments
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

Just fork and submit a PR!
📬 Feedback

Have ideas to make this simpler? Found a bug?
Open an issue or PR on GitHub!
```
