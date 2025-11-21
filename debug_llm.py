from src.llm_engine import LLMEngine
import logging
import sys

# Configure logging to see errors clearly
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def test_llm():
    print("Attempting to initialize LLMEngine...")
    try:
        engine = LLMEngine()
        if engine.model:
            print("✅ LLM Loaded Successfully!")
            print("Testing generation...")
            response = engine.analyze("Say hello")
            print(f"Response: {response}")
        else:
            print("❌ LLM Failed to Load (engine.model is None)")
    except Exception as e:
        print(f"❌ Exception during initialization: {e}")

if __name__ == "__main__":
    test_llm()
