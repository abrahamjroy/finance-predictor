import os
from pathlib import Path
from gpt4all import GPT4All
from .utils import get_logger, MODELS_DIR

logger = get_logger(__name__)

# Active Model: Phi-3 Mini (3.8B) - Best for reasoning and analysis
MODEL_NAME = "Phi-3-mini-4k-instruct-q4.gguf"

# Alternative: Qwen2.5-3B-Instruct - Faster, good generalist
# MODEL_NAME = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"

class LLMEngine:
    """
    Handles local LLM inference using GPT4All.
    """
    
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the model, checking local 'models' dir first."""
        try:
            # Check if model exists locally in the 'models' folder
            local_path = MODELS_DIR / MODEL_NAME
            
            if local_path.exists():
                logger.info(f"Found local model at {local_path}")
                # GPT4All accepts a path to the model file
                self.model = GPT4All(model_name=MODEL_NAME, model_path=str(MODELS_DIR), device='gpu', allow_download=False)
            else:
                logger.info(f"Model not found locally. Downloading {MODEL_NAME}...")
                self.model = GPT4All(MODEL_NAME, device='gpu', allow_download=True)
                
            logger.info("LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")

    def analyze(self, prompt: str) -> str:
        """Generates a response for the given prompt."""
        if not self.model:
            return "Error: LLM not loaded."
        
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
        try:
            output = self.model.generate(
                formatted_prompt, 
                max_tokens=512, 
                temp=0.7
            )
            return output
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return "Error generating analysis."
