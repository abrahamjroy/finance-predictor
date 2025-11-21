from gpt4all import GPT4All
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def test_phi3():
    model_name = "Phi-3-mini-4k-instruct.Q4_0.gguf"
    print(f"Attempting to load {model_name}...")
    try:
        # allow_download=True should trigger download if missing
        model = GPT4All(model_name, allow_download=True)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

if __name__ == "__main__":
    test_phi3()
