import os
import json
import subprocess
import sys
from pathlib import Path
from .utils import get_logger

logger = get_logger(__name__)

class LLMEngine:
    """
    Handles local LLM inference by calling an isolated subprocess.
    This prevents 'Access Violation' errors caused by library conflicts
    in the main application process.
    """

    def __init__(self):
        # No eager loading needed for subprocess approach
        # The subprocess loads the model on demand (or could be a persistent server, 
        # but for now on-demand is safer for stability)
        self.model = True # Dummy flag to satisfy app checks

    def _load_model(self):
        """
        Legacy method kept for compatibility. 
        The actual loading happens in the subprocess.
        """
        pass

    def analyze(self, prompt: str) -> str:
        """
        Generate a response by running the inference script in a subprocess.
        """
        try:
            script_path = Path(__file__).parent / "inference_script.py"
            
            # Prepare input JSON
            input_data = json.dumps({"prompt": prompt})
            
            # Run subprocess
            # We use sys.executable to ensure we use the same python environment
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            stdout, stderr = process.communicate(input=input_data)
            
            if stderr:
                logger.warning(f"LLM Subprocess Stderr: {stderr}")
                
            if process.returncode != 0:
                logger.error(f"LLM Subprocess failed with code {process.returncode}")
                return f"Error: AI process failed. Logs: {stderr}"
                
            # Parse output
            try:
                result = json.loads(stdout)
                if "error" in result:
                    return f"Error from AI: {result['error']}"
                return result.get("response", "Error: No response received.")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON output: {stdout}")
                return f"Error: Invalid output from AI process. Raw: {stdout}"

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return f"Error during analysis: {e}"

    def chat(self, messages: list) -> str:
        """
        Support for chat history. Converts messages to a single prompt string for now.
        """
        # Simple conversion for the Phi model
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            prompt += f"<|{role}|>\n{content}<|end|>\n"
        
        prompt += "<|assistant|>\n"
        return self.analyze(prompt)