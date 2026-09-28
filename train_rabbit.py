"""
Rabbit-2B Fine-Tuning Pipeline
==============================
Fine-tunes the double-abliterated MiniCPM5-2B model on the Rabbit persona,
unrestricted instruction-following, and deep technical engineering dataset using LoRA,
then merges the adapter into a standalone model and quantizes to FP8.

Output paths:
  - LoRA Adapter: ./checkpoints/Rabbit-LoRA
  - Standalone BF16 Model: ./abliterated_models/Rabbit-2B
  - Final FP8 Model: ./quantized_models/Rabbit-2B-FP8
"""

import os
import sys
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL_DIR   = "./abliterated_models/MiniCPM5-2B-DoubleAbliterated"
DATASET_PATH     = "./LlamaFactory/data/rabbit_instruct.json"
LORA_OUTPUT_DIR  = "./checkpoints/Rabbit-LoRA"
MERGED_MODEL_DIR = "./abliterated_models/Rabbit-2B"
FINAL_FP8_DIR    = "./quantized_models/Rabbit-2B-FP8"


def train_lora():
    print("=" * 65)
    print("  PHASE 1: LoRA FINE-TUNING -> Rabbit-2B")
    print(f"  Base Model : {BASE_MODEL_DIR}")
    print(f"  Dataset    : {DATASET_PATH}")
    print(f"  Output     : {LORA_OUTPUT_DIR}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/5] Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    print("[2/5] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    print("[3/5] Setting up LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print(f"[4/5] Preparing dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    def format_and_tokenize(sample):
        instruction = sample["instruction"]
        input_text = sample.get("input", "")
        output_text = sample["output"]

        user_content = f"{instruction}\n{input_text}".strip() if input_text else instruction
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text}
        ]

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        user_prompt = tokenizer.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)

        full_ids = tokenizer(full_prompt, truncation=True, max_length=1024, return_tensors=None)["input_ids"]
        user_ids = tokenizer(user_prompt, truncation=True, max_length=1024, return_tensors=None)["input_ids"]

        prompt_len = min(len(user_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    hf_dataset = Dataset.from_list(data)
    tokenized_dataset = hf_dataset.map(format_and_tokenize, remove_columns=hf_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        num_train_epochs=12,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        warmup_steps=2,
        weight_decay=0.01,
        logging_steps=2,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\n[5/5] Starting LoRA training loop...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {LORA_OUTPUT_DIR}...")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print("LoRA training complete!")
    return tokenizer


def merge_and_save():
    print("\n" + "=" * 65)
    print("  PHASE 2: MERGING LoRA WEIGHTS INTO STANDALONE MODEL")
    print(f"  Adapter : {LORA_OUTPUT_DIR}")
    print(f"  Target  : {MERGED_MODEL_DIR}")
    print("=" * 65)

    from peft import PeftModel

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Loading and attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged standalone model to {MERGED_MODEL_DIR}...")
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    print(f"Standalone Rabbit-2B model saved successfully!")


def quantize_to_fp8():
    print("\n" + "=" * 65)
    print("  PHASE 3: FP8 QUANTIZATION OF Rabbit-2B")
    print(f"  Input  : {MERGED_MODEL_DIR}")
    print(f"  Output : {FINAL_FP8_DIR}")
    print("=" * 65)

    import modelopt.torch.opt as mto
    mto.enable_huggingface_checkpointing()
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_DIR,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    sample_texts = [
        "Rabbit is a specialized systems engineering and machine learning assistant.",
        "The Linux kernel manages process scheduling, virtual memory, and hardware interrupts.",
        "CUDA thread hierarchies consist of threads, warps, thread blocks, and grid dispatch.",
        "FP8 representation scales deep neural network inference while preserving numerical stability.",
    ] * 32

    batches = [tokenizer(t, max_length=512, truncation=True, return_tensors="pt") for t in sample_texts]

    def forward_loop(mod):
        dev = next(mod.parameters()).device
        with torch.no_grad():
            for i, b in enumerate(batches):
                mod(input_ids=b["input_ids"].to(dev))
                if (i + 1) % 32 == 0:
                    print(f"  Calibrated {i + 1}/{len(batches)} samples")

    print("Quantizing with NVIDIA ModelOpt FP8...")
    quantized = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    os.makedirs(FINAL_FP8_DIR, exist_ok=True)
    with torch.inference_mode():
        export_hf_checkpoint(quantized, export_dir=FINAL_FP8_DIR)
        tokenizer.save_pretrained(FINAL_FP8_DIR)

    print(f"\nFinal Rabbit-2B-FP8 exported to: {FINAL_FP8_DIR}")


if __name__ == "__main__":
    train_lora()
    merge_and_save()
    quantize_to_fp8()
    print("\n" + "=" * 65)
    print("  ALL PHASES COMPLETE: Rabbit-2B is trained, merged, and FP8 quantized!")
    print("=" * 65)
