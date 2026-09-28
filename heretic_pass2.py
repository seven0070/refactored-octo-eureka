"""
Heretic Pass 2 + FP8 Re-quantize
==================================
Takes the already-OBLITERATUS-abliterated MiniCPM5-2B BF16 model
and runs Heretic (Bayesian-optimized abliteration) as a second pass,
then re-quantizes the result to FP8.

Expected result: ~99%+ refusal removal (vs 96.7% after pass 1 alone)

Output: ./quantized_models/MiniCPM5-2B-DoubleAbliterated-FP8
"""

import os
import sys
import torch
import subprocess

ABLITERATED_DIR     = "./abliterated_models/MiniCPM5-2B-Abliterated"
HERETIC_OUT_DIR     = "./abliterated_models/MiniCPM5-2B-DoubleAbliterated"
FINAL_FP8_DIR       = "./quantized_models/MiniCPM5-2B-DoubleAbliterated-FP8"


def step1_heretic_pass2():
    """
    Run Heretic CLI on the already-OBLITERATUS-abliterated BF16 model.
    Uses Bayesian/TPE optimization (Optuna) to find parameters that
    simultaneously minimize: refusal_rate and KL_divergence.
    """
    print("\n" + "=" * 65)
    print("  HERETIC — PASS 2 ABLITERATION (Bayesian-optimized)")
    print(f"  Input  : {ABLITERATED_DIR}")
    print(f"  Output : {HERETIC_OUT_DIR}")
    print("=" * 65)

    import shutil
    heretic_exe = shutil.which("heretic") or shutil.which("heretic.exe")
    if not heretic_exe:
        print("\n[ERROR] heretic CLI not found.")
        print("  Run: pip install -U heretic-llm")
        sys.exit(1)

    os.makedirs(HERETIC_OUT_DIR, exist_ok=True)
    abs_input  = os.path.abspath(ABLITERATED_DIR)
    abs_output = os.path.abspath(HERETIC_OUT_DIR)

    cmd = [
        heretic_exe,
        "--model", abs_input,
        "--n-trials", "30",               # Bayesian search trials
        "--kl-divergence-target", "0.3",  # tight KL budget = less capability loss
        "--export-strategy", "MERGE",     # save merged weights (not LoRA adapter)
    ]

    print(f"\n  Command: {' '.join(cmd)}\n")
    print("  Heretic will:")
    print("    1. Benchmark GPU for optimal batch size")
    print("    2. Run 30 Bayesian (TPE) trials")
    print("    3. Co-minimize: refusal_rate + KL_divergence")
    print("    4. Prompt to save the best model\n")
    print("  NOTE: When Heretic asks where to save, enter:")
    print(f"    {abs_output}\n")

    # Force UTF-8 so Heretic's rich/emoji output doesn't crash on Windows cp1252
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Run with live output so user sees Heretic's interactive progress
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"\n[WARNING] Heretic exited with code {result.returncode}")
        print("  The Heretic output may still be usable if it saved before exiting.")
    else:
        print("\n  Heretic pass 2 complete!")

    return HERETIC_OUT_DIR


def step2_requantize_fp8(input_dir: str):
    """Re-quantize the double-abliterated BF16 model to FP8."""
    print("\n" + "=" * 65)
    print("  STEP 2: FP8 RE-QUANTIZATION")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {FINAL_FP8_DIR}")
    print("=" * 65)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import modelopt.torch.opt as mto
    mto.enable_huggingface_checkpointing()
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        input_dir, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Loading double-abliterated BF16 model...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        input_dir,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()

    print("  Building 128-sample calibration set...")
    texts = [
        "The history of artificial intelligence spans decades of research.",
        "Quantum computing uses superposition to solve problems exponentially faster.",
        "The human genome contains three billion base pairs of DNA.",
        "Neural networks are trained with gradient descent and backpropagation.",
        "Transformers use self-attention to process sequences in parallel.",
        "Protein folding determines molecular function in biological systems.",
        "The speed of light is approximately 299,792,458 meters per second.",
        "Semiconductor fabrication operates at nanometer-scale precision.",
    ] * 16  # 128 samples

    batches = [
        tokenizer(t, max_length=512, truncation=True, padding=False, return_tensors="pt")
        for t in texts[:128]
    ]

    def forward_loop(model):
        dev = next(model.parameters()).device
        with torch.no_grad():
            for i, b in enumerate(batches):
                ids = b["input_ids"].to(dev)
                mask = b.get("attention_mask")
                model(input_ids=ids, attention_mask=mask.to(dev) if mask is not None else None)
                if (i + 1) % 32 == 0 or (i + 1) == len(batches):
                    print(f"    Calibrated {i + 1}/{len(batches)}")

    print("  Quantizing to FP8...")
    quantized = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    os.makedirs(FINAL_FP8_DIR, exist_ok=True)
    print(f"\n  Exporting to {FINAL_FP8_DIR}...")
    with torch.inference_mode():
        export_hf_checkpoint(quantized, export_dir=FINAL_FP8_DIR)
        tokenizer.save_pretrained(FINAL_FP8_DIR)

    size_gb = sum(
        os.path.getsize(os.path.join(FINAL_FP8_DIR, f))
        for f in os.listdir(FINAL_FP8_DIR)
        if os.path.isfile(os.path.join(FINAL_FP8_DIR, f))
    ) / (1024 ** 3)

    print("\n" + "=" * 65)
    print("  DOUBLE-ABLITERATION PIPELINE COMPLETE")
    print(f"  OBLITERATUS pass 1 output : {ABLITERATED_DIR}")
    print(f"  Heretic pass 2 output     : {HERETIC_OUT_DIR}")
    print(f"  Final FP8 model           : {FINAL_FP8_DIR}")
    print(f"  Final model size          : {size_gb:.2f} GB")
    print("\n  Load with vLLM:")
    print(f'    llm = LLM(model=r"{FINAL_FP8_DIR}", quantization="fp8")')
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Heretic Pass 2 + FP8 Quantization")
    parser.add_argument("--skip-heretic", action="store_true", help="Skip Heretic step and directly quantize the existing model in HERETIC_OUT_DIR")
    args = parser.parse_args()

    if getattr(args, "skip_heretic", False):
        heretic_output = HERETIC_OUT_DIR
        print(f"Skipping Heretic step, using existing model at: {heretic_output}")
    else:
        heretic_output = step1_heretic_pass2()

    # Only proceed to FP8 quantization if Heretic saved a model
    any_safetensors = any(
        f.endswith(".safetensors")
        for f in os.listdir(heretic_output)
        if os.path.isfile(os.path.join(heretic_output, f))
    ) if os.path.isdir(heretic_output) else False

    if not os.path.isdir(heretic_output) or not any_safetensors:
        print(f"\n[ERROR] Heretic output not found or contains no .safetensors files at {heretic_output}")
        print("  Please make sure Heretic saved the model into that folder.")
        print(f"  Target path: {os.path.abspath(heretic_output)}")
        print(f"  Then run: python heretic_pass2.py --skip-heretic")
        sys.exit(1)

    step2_requantize_fp8(heretic_output)
