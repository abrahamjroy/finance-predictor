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
        self.model = True  # Dummy flag to satisfy app checks

    def _load_model(self):
        """Legacy method kept for compatibility."""
        pass

    def analyze(self, prompt: str, stream_callback=None) -> dict:
        """
        Generate a response by running the inference script in a subprocess.
        If stream_callback is provided, it will be called with each chunk.
        Returns dict with 'response', 'thinking', and 'strategy' keys.
        """
        try:
            script_path = Path(__file__).parent / "inference_script.py"
            
            # Prepare input JSON
            input_data = json.dumps({"prompt": prompt})
            
            # Run subprocess
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                bufsize=1  # Line buffered for streaming
            )
            
            # Send input
            process.stdin.write(input_data)
            process.stdin.close()
            
            # Read streaming output
            full_output = ""
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    chunk_data = json.loads(line)
                    if "chunk" in chunk_data and stream_callback:
                        stream_callback(chunk_data["chunk"])
                    full_output += line + "\n"
                except json.JSONDecodeError:
                    full_output += line + "\n"
            
            stderr = process.stderr.read()
            process.wait()
            
            if stderr:
                # Filter out benign warnings from llama.cpp
                if "n_ctx_per_seq" in stderr and "full capacity" in stderr:
                    pass # Ignore this specific warning
                elif "llama_kv_cache_unified" in stderr:
                    pass # Ignore V cache warning
                else:
                    logger.warning(f"LLM Subprocess Stderr: {stderr}")
                
            if process.returncode != 0:
                logger.error(f"LLM Subprocess failed with code {process.returncode}")
                return {"error": f"AI process failed. Logs: {stderr}"}
                
            # Parse final output (last line should be the result JSON)
            lines = [l.strip() for l in full_output.split("\n") if l.strip()]
            for line in reversed(lines):
                try:
                    result = json.loads(line)
                    if "error" in result or "response" in result:
                        return result
                except json.JSONDecodeError:
                    continue
                    
            return {"error": "No valid response received"}

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return {"error": f"Error during analysis: {e}"}

    def chat(self, messages: list) -> str:
        """Support for chat history."""
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            prompt += f"<|{role}|>\n{content}<|end|>\n"
        
        prompt += "<|assistant|>\n"
        result = self.analyze(prompt)
        return result.get("response", str(result))