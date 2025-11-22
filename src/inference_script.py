import sys
import os
import json
import multiprocessing
from pathlib import Path

# Ensure we can import from src if needed, but we'll try to keep this self-contained
# or assume run from project root.

def get_model_path():
    # Assuming script is run from project root
    return Path("models") / "Phi-4-mini-reasoning-Q4_K_M.gguf"

def get_optimal_threads():
    try:
        cpu_count = multiprocessing.cpu_count()
        return max(4, int(cpu_count * 0.75))
    except:
        return 4

def load_model_and_infer(prompt):
    model_path = get_model_path()
    if not model_path.exists():
        return {"error": f"Model not found at {model_path}"}

    # Strategies
    strategies = [
        (-1, "Max GPU Acceleration"),
        (20, "Hybrid GPU/CPU (Safe Mode)"),
        (0, "CPU Only (CUDA Disabled)"),
    ]

    from llama_cpp import Llama

    model = None
    used_strategy = None

    for layers, mode in strategies:
        try:
            # Set/Unset CUDA env var based on strategy
            if layers == 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            else:
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]

            # Attempt load
            model = Llama(
                model_path=str(model_path),
                n_gpu_layers=layers,
                n_ctx=4096,
                n_threads=get_optimal_threads(),
                verbose=False, # Keep stderr clean
                model_type="phi"
            )
            used_strategy = mode
            break # Success
        except Exception as e:
            # Log to stderr so it doesn't pollute stdout JSON
            print(f"Strategy {mode} failed: {e}", file=sys.stderr)
            continue

    if model is None:
        return {"error": "All loading strategies failed."}

    # Inference
    messages = [
        {"role": "system", "content": "You are a helpful financial analyst assistant."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=2048, # Increased to prevent cut-offs
            temperature=0.7,
            top_p=0.9,
        )
        content = response["choices"][0]["message"]["content"].strip()
        
        # Post-processing: Remove <think>...</think> blocks
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        return {"response": content, "strategy": used_strategy}
    except Exception as e:
        return {"error": f"Inference failed: {e}"}

def main():
    try:
        # Read input from stdin (expecting JSON)
        input_str = sys.stdin.read()
        if not input_str.strip():
            # Fallback for testing: check args
            if len(sys.argv) > 1:
                prompt = sys.argv[1]
            else:
                print(json.dumps({"error": "No input provided"}))
                return
        else:
            try:
                data = json.loads(input_str)
                prompt = data.get("prompt", "")
            except:
                # Treat raw input as prompt
                prompt = input_str

        if not prompt:
            print(json.dumps({"error": "Empty prompt"}))
            return

        result = load_model_and_infer(prompt)
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": f"Script error: {e}"}))

if __name__ == "__main__":
    main()
