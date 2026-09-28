"""
Full Pipeline: Abliterate → FP8 Quantize
=========================================
Step 1: Run OBLITERATUS abliteration on the base MiniCPM5-2B (BF16) model
         → removes refusal directions via SVD weight surgery
Step 2: FP8 quantize the abliterated BF16 model using NVIDIA Model Optimizer
         → final output: MiniCPM5-2B-Abliterated-FP8 (~2.84 GB)

Usage:
    python abliterate_then_quantize.py
    python abliterate_then_quantize.py --skip-abliterate  # resume from existing
    python abliterate_then_quantize.py --method surgical   # change abliteration method
"""

import argparse
import os
import sys
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL_ID   = "openbmb/MiniCPM5-2B"
ABLITERATED_DIR = "./abliterated_models/MiniCPM5-2B-Abliterated"
FINAL_FP8_DIR   = "./quantized_models/MiniCPM5-2B-Abliterated-FP8"


def step1_abliterate(method: str = "advanced"):
    """
    Step 1: Abliterate the base BF16 model using OBLITERATUS.
    Surgically removes the refusal subspace from model weights via SVD.
    """
    print("\n" + "=" * 65)
    print("  STEP 1: OBLITERATUS ABLITERATION")
    print(f"  Model  : {BASE_MODEL_ID}")
    print(f"  Method : {method}")
    print(f"  Output : {ABLITERATED_DIR}")
    print("=" * 65)

    try:
        from obliteratus.abliterate import AbliterationPipeline
    except ImportError:
        print("\n[ERROR] OBLITERATUS is not installed.")
        print("  Run: pip install -e OBLITERATUS/")
        sys.exit(1)

    os.makedirs(ABLITERATED_DIR, exist_ok=True)

    pipeline = AbliterationPipeline(
        model_name=BASE_MODEL_ID,
        method=method,
        output_dir=ABLITERATED_DIR,
        max_seq_length=512,
        trust_remote_code=True,
        dtype="bfloat16",   # MiniCPM5-2B is a BF16 model
    )

    print("\n  Stages: SUMMON -> PROBE -> DISTILL -> EXCISE -> VERIFY -> REBIRTH\n")
    result = pipeline.run()

    print("\n  Abliteration complete!")
    if hasattr(result, "metrics") and result.metrics:
        m = result.metrics
        print(f"    Perplexity:   {m.get('perplexity', 'N/A')}")
        print(f"    Refusal rate: {m.get('refusal_rate', 'N/A')}")
        print(f"    Coherence:    {m.get('coherence', 'N/A')}")

    return ABLITERATED_DIR


def step2_quantize_fp8(abliterated_model_path: str):
    """
    Step 2: FP8 quantize the abliterated BF16 model using NVIDIA Model Optimizer.
    """
    print("\n" + "=" * 65)
    print("  STEP 2: NVIDIA MODEL OPTIMIZER — FP8 QUANTIZATION")
    print(f"  Input  : {abliterated_model_path}")
    print(f"  Output : {FINAL_FP8_DIR}")
    print("=" * 65)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    import modelopt.torch.opt as mto
    mto.enable_huggingface_checkpointing()
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  WARNING: No CUDA detected — running on CPU (very slow).")

    # Load tokenizer
    print("\n  [2a/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        abliterated_model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load abliterated BF16 model
    print("  [2b/4] Loading abliterated BF16 model...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        abliterated_model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()

    # Build calibration data
    print("  [2c/4] Preparing calibration data (128 samples)...")
    calib_seqlen = 512
    sample_texts = [
        "Artificial intelligence research has advanced rapidly with transformer-based models.",
        "The human genome contains approximately three billion base pairs of DNA.",
        "Modern neural networks are trained using gradient descent and backpropagation.",
        "Large language models generate text by predicting the next token in a sequence.",
        "Quantum computing leverages superposition and entanglement for computation.",
        "The laws of thermodynamics govern energy transfer in all physical systems.",
        "Machine learning models require diverse training data to generalize effectively.",
        "Transformer architectures use self-attention to process sequences in parallel.",
        "Semiconductor fabrication involves photolithography at nanometer scales.",
        "The universe is estimated to be approximately 13.8 billion years old.",
        "Ocean acidification threatens coral reef ecosystems and marine biodiversity.",
        "Protein folding determines molecular function and is central to drug discovery.",
        "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "Immunotherapy has revolutionized cancer treatment by harnessing the immune system.",
        "Natural language processing enables computers to understand and generate human text.",
    ] * 8  # 16 * 8 = 128 samples

    batches = []
    for text in sample_texts[:128]:
        tokens = tokenizer(
            text,
            max_length=calib_seqlen,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        batches.append(tokens)

    def forward_loop(model):
        print("  Running FP8 calibration forward pass...")
        dev = next(model.parameters()).device
        with torch.no_grad():
            for i, batch in enumerate(batches):
                input_ids = batch["input_ids"].to(dev)
                mask = batch.get("attention_mask")
                if mask is not None:
                    model(input_ids=input_ids, attention_mask=mask.to(dev))
                else:
                    model(input_ids=input_ids)
                if (i + 1) % 32 == 0 or (i + 1) == len(batches):
                    print(f"    Calibrated {i + 1}/{len(batches)} samples")

    # FP8 quantize
    print("  [2d/4] Applying FP8 quantization (mtq.FP8_DEFAULT_CFG)...")
    quantized_model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    # Export checkpoint
    os.makedirs(FINAL_FP8_DIR, exist_ok=True)
    print(f"\n  Exporting HF checkpoint to {FINAL_FP8_DIR}...")
    with torch.inference_mode():
        export_hf_checkpoint(quantized_model, export_dir=FINAL_FP8_DIR)
        tokenizer.save_pretrained(FINAL_FP8_DIR)

    # Report final size
    total_bytes = sum(
        os.path.getsize(os.path.join(FINAL_FP8_DIR, f))
        for f in os.listdir(FINAL_FP8_DIR)
        if os.path.isfile(os.path.join(FINAL_FP8_DIR, f))
    )
    size_gb = total_bytes / (1024 ** 3)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Abliterated BF16 : {ABLITERATED_DIR}")
    print(f"  Final FP8 model  : {FINAL_FP8_DIR}")
    print(f"  Final model size : {size_gb:.2f} GB")
    print("\n  Load with vLLM:")
    print(f'    from vllm import LLM')
    print(f'    llm = LLM(model=r"{FINAL_FP8_DIR}", quantization="fp8")')
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Abliterate + FP8 quantize MiniCPM5-2B pipeline"
    )
    parser.add_argument(
        "--method",
        default="advanced",
        choices=["basic", "advanced", "surgical", "informed", "optimized"],
        help="OBLITERATUS abliteration method (default: advanced)"
    )
    parser.add_argument(
        "--skip-abliterate",
        action="store_true",
        help="Skip Step 1, use existing abliterated model in ABLITERATED_DIR"
    )
    args = parser.parse_args()

    if args.skip_abliterate:
        if not os.path.isdir(ABLITERATED_DIR):
            print(f"[ERROR] --skip-abliterate set but {ABLITERATED_DIR} does not exist.")
            sys.exit(1)
        print(f"  Skipping abliteration. Using: {ABLITERATED_DIR}")
        abliterated_path = ABLITERATED_DIR
    else:
        abliterated_path = step1_abliterate(method=args.method)

    step2_quantize_fp8(abliterated_path)


if __name__ == "__main__":
    main()
