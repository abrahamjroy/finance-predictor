import sys
sys.path.insert(0, 'c:/Users/royka/.gemini/antigravity/scratch/finance_predictor')
from src.llm_engine import LLMEngine

engine = LLMEngine()
print('Model type:', type(engine.model))
if hasattr(engine.model, 'create_chat_completion'):
    try:
        resp = engine.analyze('What is 2+2?')
        print('Response:', resp)
    except Exception as e:
        print('Analysis exception:', e)
else:
    print('No valid model loaded.')
