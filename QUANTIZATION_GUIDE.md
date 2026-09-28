# NVIDIA Model Optimizer - Quantization Guide for MiniCPM5-2B

This guide outlines how to quantize **`openbmb/MiniCPM5-2B`** (and other Hugging Face models) using **NVIDIA Model Optimizer (`nvidia-modelopt`)** into high-performance **FP8** (or INT4 AWQ) on your system.

---

## 1. System & Environment Specifications

- **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM)
- **CUDA Driver**: 13.3 (Driver version 610.88)
- **PyTorch**: `2.8.0+cu126` (CUDA enabled)
- **Model Optimizer**: `nvidia-modelopt 0.47.0` (with Hugging Face unified checkpoint export)
- **Model Architecture**: `openbmb/MiniCPM5-2B` (`LlamaForCausalLM` architecture, 2.5B dense parameters, 128k context)

---

## 2. Quantization Script Overview

A dedicated script has been created in your workspace:
[`quantize_minicpm.py`](./quantize_minicpm.py)

### Key Features:
- **Formats Supported**:
  - `fp8`: W8A8 FP8 (E4M3 format) for optimal throughput on Ada/Blackwell architecture Tensor Cores.
  - `int4_awq`: Activation-aware Weight Quantization (4-bit weights) for maximum memory savings.
  - `int8_sq`: SmoothQuant W8A8 INT8.
- **Export Format**: Hugging Face Unified Checkpoint (compatible with vLLM, SGLang, and TensorRT-LLM).
- **Calibration**: Automatically calibrates dynamic activation ranges using `cnn_dailymail` / `wikitext` with built-in offline fallbacks.

---

## 3. Running Quantization

Open PowerShell and execute:

### FP8 Quantization (Recommended)
```powershell
& 'C:\Users\sanath\AppData\Local\Programs\Python\Python311\python.exe' quantize_minicpm.py `
  --model openbmb/MiniCPM5-2B `
  --format fp8 `
  --calib-samples 128 `
  --calib-seqlen 512 `
  --output ./quantized_models/MiniCPM5-2B-FP8
```

### INT4 AWQ Quantization (Maximum VRAM Compression)
```powershell
& 'C:\Users\sanath\AppData\Local\Programs\Python\Python311\python.exe' quantize_minicpm.py `
  --model openbmb/MiniCPM5-2B `
  --format int4_awq `
  --calib-samples 128 `
  --calib-seqlen 512 `
  --output ./quantized_models/MiniCPM5-2B-INT4-AWQ
```

---

## 4. Serving the Quantized Checkpoint

Once exported to `./quantized_models/MiniCPM5-2B-FP8`, the directory contains:
- `config.json` with the embedded `quantization_config`
- `model.safetensors` in FP8
- `tokenizer.json` and tokenizer configurations

### Deploying with vLLM
```bash
vllm serve ./quantized_models/MiniCPM5-2B-FP8 --dtype auto --kv-cache-dtype fp8
```

### Loading Directly in Hugging Face Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./quantized_models/MiniCPM5-2B-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

inputs = tokenizer("Artificial Intelligence is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 5. Official NVIDIA Model-Optimizer CLI Examples

The full repository is cloned in [`./Model-Optimizer`](./Model-Optimizer). You can also run the official recipes directly:

```powershell
cd Model-Optimizer/examples/hf_ptq
& 'C:\Users\sanath\AppData\Local\Programs\Python\Python311\python.exe' hf_ptq.py `
  --pyt_ckpt_path openbmb/MiniCPM5-2B `
  --recipe general/ptq/fp8_default-kv_fp8_cast `
  --export_path ../../../quantized_models/MiniCPM5-2B-FP8-official
```
