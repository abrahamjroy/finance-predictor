import os
from ctransformers import AutoModelForCausalLM

# Get absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "Phi-4-mini-reasoning-Q4_K_M.gguf")

print(f"Attempting to load {model_path} with ctransformers...")

try:
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, 
        model_type="phi2", # Phi-4 is likely based on Phi-2/3 architecture
        gpu_layers=0
    )
    print("✅ Model loaded successfully!")
    print(llm("The future of AI is"))
except Exception as e:
    print(f"❌ Failed: {e}")
