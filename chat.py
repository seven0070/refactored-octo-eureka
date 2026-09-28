import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

sys.stdout.reconfigure(encoding="utf-8")
path = r"d:\RAD\model optimizer\abliterated_models\Rabbit-2B"

print("Loading Rabbit-2B...")
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
mod = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

history = []
print("\n--- Rabbit-2B Ready. Type 'exit' to quit. ---")
while True:
    try:
        user_input = input("\nYou > ").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            break
        history.append({"role": "user", "content": user_input})
        prompt = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        print("\nRabbit > ", end="", flush=True)
        with torch.inference_mode():
            out = mod.generate(**inputs, streamer=streamer, max_new_tokens=512, temperature=0.6)
        history.append({"role": "assistant", "content": tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)})
    except KeyboardInterrupt:
        break
