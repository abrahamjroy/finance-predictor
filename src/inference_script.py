import sys
import os
import json
import multiprocessing
from pathlib import Path

# Ensure we can import from src if needed, but we'll try to keep this self-contained
# or assume run from project root.

def get_model_path():
    # Assuming script is run from project root
    return Path("models") / "granite-4.0-h-tiny-adaptive-reasoning.i1-IQ4_XS.gguf"

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
                n_ctx=16384, 
                n_threads=get_optimal_threads(),
                verbose=False, 
                flash_attn=True # Enable Flash Attention for Granite
            )
            used_strategy = mode
            break # Success
        except Exception as e:
            # Log to stderr so it doesn't pollute stdout JSON
            print(f"Strategy {mode} failed: {e}", file=sys.stderr)
            continue

    if model is None:
        return {"error": "All loading strategies failed."}

    # Inference with streaming
    messages = [
        {"role": "system", "content": """You are an autonomous financial agent.
Your goal is to assist the user by controlling the application using specific JSON commands.

TOOLS AVAILABLE:
- load_ticker(symbol): Load a stock. JSON: {"tool": "load_ticker", "params": {"symbol": "TICKER"}}
- run_predictions(): Run models. JSON: {"tool": "run_predictions"}
- set_forecast_days(days): Set horizon. JSON: {"tool": "set_forecast_days", "params": {"days": 30}}
- show_chart_indicator(indicator): Toggle indicator. JSON: {"tool": "show_chart_indicator", "params": {"indicator": "SMA"}}

CRITICAL INSTRUCTION:
You are a Reasoning Model. You will naturally think about the problem first.
That is fine, BUT YOU MUST END YOUR RESPONSE WITH THE JSON COMMANDS.
If you do not output the JSON block, the application will do nothing.

FORMAT:
[Optional: Your reasoning text...]

```json
[
  {"tool": "load_ticker", "params": {"symbol": "NVDA"}},
  {"tool": "set_forecast_days", "params": {"days": 15}},
  {"tool": "run_predictions"}
]
```
"""},
        {"role": "user", "content": prompt},
    ]

    try:
        # Speed optimizations: lower max_tokens, temperature, enable streaming
        thinking_content = ""
        final_content = ""
        
        for chunk in model.create_chat_completion(
            messages=messages,
            max_tokens=1024,  # Reduced for speed
            temperature=0.5,  # Lower for faster, more focused responses
            top_p=0.85,
            stream=True  # Enable streaming
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content_chunk = delta.get("content", "")
            if content_chunk:
                # Output streaming chunks
                print(json.dumps({"chunk": content_chunk}), flush=True)
                final_content += content_chunk
        
        # Post-processing: Extract thinking and final answer
        import re
        thinking_match = re.search(r'<think>(.*?)</think>', final_content, re.DOTALL)
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            final_answer = re.sub(r'<think>.*?</think>', '', final_content, flags=re.DOTALL).strip()
        else:
            final_answer = final_content.strip()
        
        return {
            "response": final_answer,
            "thinking": thinking_content,
            "strategy": used_strategy
        }
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
