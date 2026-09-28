"""
NVIDIA Model Optimizer (modelopt) - FP8 Post-Training Quantization (PTQ) Script
Tailored for: openbmb/MiniCPM5-2B (Llama architecture)
Target precision: FP8 (E4M3 weights and activations) with optional FP8 KV-Cache
Output format: Hugging Face Unified Checkpoint (compatible with vLLM, SGLang, and TensorRT-LLM)
"""

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Enable HuggingFace checkpoint export support for ModelOpt
import modelopt.torch.opt as mto
mto.enable_huggingface_checkpointing()
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def get_calibration_dataloader(tokenizer, num_samples=128, seq_len=512, dataset_name="cnn_dailymail"):
    """
    Creates a simple calibration dataloader using a standard text dataset with local fallback.
    """
    batches = []
    print(f"Loading calibration dataset '{dataset_name}' ({num_samples} samples)...")
    try:
        if dataset_name == "cnn_dailymail":
            ds = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{num_samples * 2}]")
            text_column = "article"
        elif dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{num_samples * 2}]")
            text_column = "text"
        else:
            ds = load_dataset(dataset_name, split=f"train[:{num_samples * 2}]")
            text_column = "text"

        for item in ds:
            text = item[text_column].strip()
            if not text:
                continue
            tokens = tokenizer(
                text,
                max_length=seq_len,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )
            if tokens["input_ids"].shape[1] > 16:
                batches.append(tokens)
                if len(batches) >= num_samples:
                    break
    except Exception as e:
        print(f"Warning: Could not fetch online dataset ({e}). Using built-in diverse calibration text samples.")
        sample_texts = [
            "The NVIDIA Model Optimizer enables state-of-the-art post-training quantization and pruning algorithms for large language models.",
            "Artificial intelligence research has progressed rapidly with transformer-based deep neural networks across language and vision.",
            "Quantization is a technique that maps floating-point numbers to lower precision representations such as FP8 and INT4.",
            "In deep learning, activation scaling factors are calibrated by running sample batches through the network forward loop.",
            "The RTX GPU architecture incorporates specialized Tensor Cores capable of high-throughput FP8 and INT4 matrix operations.",
            "MiniCPM is a lightweight language model designed for efficient on-device intelligence and edge deployment.",
            "Post-training quantization dramatically reduces memory footprint while preserving task accuracy and perplexity.",
            "Hugging Face transformers provide a unified framework for fine-tuning, evaluating, and serving foundation models."
        ] * (num_samples // 8 + 1)
        for text in sample_texts[:num_samples]:
            tokens = tokenizer(
                text,
                max_length=seq_len,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )
            batches.append(tokens)

    print(f"Loaded {len(batches)} valid calibration sequences.")
    return batches


def quantize_model(
    model_id="openbmb/MiniCPM5-2B",
    output_dir="./quantized_models/MiniCPM5-2B-FP8",
    qformat="fp8",
    calib_samples=128,
    calib_seqlen=512,
    device="cuda",
    batch_size=1
):
    print("=" * 60)
    print(f"Starting NVIDIA Model Optimizer Quantization")
    print(f"Model ID:       {model_id}")
    print(f"Format:         {qformat}")
    print(f"Export Path:    {output_dir}")
    print(f"Device:         {device}")
    print("=" * 60)

    if not torch.cuda.is_available() and device == "cuda":
        print("WARNING: CUDA is not available. Running on CPU (may be slow).")
        device = "cpu"

    # 1. Load Tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    print(f"\n[2/4] Loading base model '{model_id}'...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()

    # 3. Prepare Calibration Forward Loop
    print("\n[3/4] Preparing calibration data...")
    calib_data = get_calibration_dataloader(
        tokenizer,
        num_samples=calib_samples,
        seq_len=calib_seqlen
    )

    def forward_loop(model):
        print("Running calibration forward loop...")
        dev = getattr(model, "device", None) or next(model.parameters()).device
        with torch.no_grad():
            for i, batch in enumerate(calib_data):
                input_ids = batch["input_ids"].to(dev)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(dev)
                    model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    model(input_ids=input_ids)
                if (i + 1) % 16 == 0 or (i + 1) == len(calib_data):
                    print(f"  Processed {i + 1}/{len(calib_data)} calibration samples")

    # 4. Quantize
    print(f"\n[4/4] Applying {qformat.upper()} quantization with ModelOpt...")
    if qformat == "fp8":
        config = mtq.FP8_DEFAULT_CFG
    elif qformat == "int4_awq":
        config = mtq.INT4_AWQ_CFG
    elif qformat == "w4a8_awq":
        config = mtq.W4A8_AWQ_BETA_CFG
    elif qformat == "nvfp4":
        config = mtq.NVFP4_DEFAULT_CFG
    elif qformat == "nvfp4_mlp_only":
        config = mtq.NVFP4_MLP_ONLY_CFG
    elif qformat == "int8_sq":
        config = mtq.INT8_SMOOTHQUANT_CFG
    else:
        raise ValueError(f"Unsupported format: {qformat}")

    # In-place quantization and calibration
    quantized_model = mtq.quantize(model, config, forward_loop)

    # 5. Export Checkpoint
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExporting quantized model to {output_dir}...")
    with torch.inference_mode():
        export_hf_checkpoint(quantized_model, export_dir=output_dir)
        tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print(f"Quantization complete! Model saved to: {output_dir}")
    print("This checkpoint can now be served with vLLM, SGLang, or TensorRT-LLM.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize LLM with NVIDIA Model Optimizer")
    parser.add_argument("--model", type=str, default="openbmb/MiniCPM5-2B", help="HF model name or local path")
    parser.add_argument("--output", type=str, default="./quantized_models/MiniCPM5-2B-FP8", help="Output export path")
    parser.add_argument("--format", type=str, default="fp8", choices=["fp8", "int4_awq", "w4a8_awq", "nvfp4", "nvfp4_mlp_only", "int8_sq"], help="Quantization format")
    parser.add_argument("--calib-samples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--calib-seqlen", type=int, default=512, help="Sequence length for calibration")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    args = parser.parse_args()

    quantize_model(
        model_id=args.model,
        output_dir=args.output,
        qformat=args.format,
        calib_samples=args.calib_samples,
        calib_seqlen=args.calib_seqlen,
        device=args.device
    )
