import sys
import os
import json
import multiprocessing
from pathlib import Path

# Ensure we can import from src if needed, but we'll try to keep this self-contained
# or assume run from project root.


def get_model_path():
    # DeepSeek-R1-Distill-Qwen-1.5B - Optimized for reasoning and financial analytics
    # Download from: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
    # Recommended quantization: Q6_K for quality or Q4_K_M for performance
    return Path("models") / "DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf"

def download_model_if_needed():
    """
    Download the model from Hugging Face if it doesn't exist locally.
    Returns the path to the model file.
    """
    model_path = get_model_path()
    
    # Check if model already exists
    if model_path.exists():
        print(f"Model found at {model_path}", file=sys.stderr)
        return model_path
    
    # Model doesn't exist, download it
    print("=" * 60, file=sys.stderr)
    print("DeepSeek-R1 model not found. Downloading automatically...", file=sys.stderr)
    print("This is a one-time download (~1.4GB for Q6_K quantization)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Create models directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the model
        downloaded_path = hf_hub_download(
            repo_id="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            filename="DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf",
            local_dir=model_path.parent,
            local_dir_use_symlinks=False
        )
        
        print(f"✓ Model downloaded successfully to {model_path}", file=sys.stderr)
        return Path(downloaded_path)
        
    except ImportError:
        error_msg = """
ERROR: huggingface_hub is not installed.
Please install it with: pip install huggingface_hub

Or download the model manually from:
https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF

Place the file at: {model_path}
""".format(model_path=model_path)
        print(error_msg, file=sys.stderr)
        return None
        
    except Exception as e:
        error_msg = f"""
ERROR: Failed to download model: {e}

Please download manually from:
https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF

Recommended file: DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf
Place it at: {model_path}
"""
        print(error_msg, file=sys.stderr)
        return None

def get_optimal_threads():
    try:
        cpu_count = multiprocessing.cpu_count()
        return max(4, int(cpu_count * 0.75))
    except:
        return 4

def load_model_and_infer(prompt):
    # Download model if needed
    model_path = download_model_if_needed()
    
    if model_path is None or not model_path.exists():
        return {"error": f"Model not found and automatic download failed. See instructions above."}

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
                n_ctx=16384,  # DeepSeek-R1 supports up to 131K context, 16K is safe
                n_threads=get_optimal_threads(),
                verbose=False
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
