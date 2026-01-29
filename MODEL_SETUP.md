# DeepSeek-R1-Distill-Qwen-1.5B Model Setup

## Overview

Finance Predictor uses **DeepSeek-R1-Distill-Qwen-1.5B**, a compact reasoning model optimized for financial analytics and structured reasoning tasks.

**Key Benefits:**
- **Smaller Size**: 1.5B parameters (vs 7B+ for Granite 4.0)
- **Faster Inference**: Optimized reasoning pipeline with Chain of Thought (CoT)
- **Financial Focus**: Specifically designed for stock analysis and SEC filing interpretation
- **Better Reasoning**: Distilled from DeepSeek-R1 with enhanced reasoning capabilities

## Download Instructions

### Option 1: Direct Download (Recommended)

Visit [bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) and download one of:

**Recommended for Financial Analysis:**
- `DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf` (1.4GB) - Best quality
- `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` (990MB) - Good balance
- `DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf` (880MB) - Fastest

### Option 2: Hugging Face CLI

```bash
# Install Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Download Q6_K (best quality)
huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf --local-dir models --local-dir-use-symlinks False

# Or download Q4_K_M (balanced)
huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf --local-dir models --local-dir-use-symlinks False
```

### Option 3: Alternative Repositories

Other GGUF sources:
- [tensorblock/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/tensorblock/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)
- [ggml-org/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF](https://huggingface.co/ggml-org/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF)

## File Placement

Place the downloaded `.gguf` file in the `models/` directory:

```
finance-predictor/
└── models/
    └── DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf  # Or your chosen quantization
```

**Note:** If you download a different quantization (Q4_K_M, Q4_0, etc.), update the filename in `src/inference_script.py` line 13.

## Quantization Guide

| Quantization | Size | Quality | Speed | Use Case |
|--------------|------|---------|-------|----------|
| **Q6_K** | 1.4GB | Excellent | Medium | Financial analysis, complex reasoning |
| **Q4_K_M** | 990MB | Good | Fast | General use, quick analysis |
| **Q4_0** | 880MB | Fair | Fastest | Testing, simple queries |
| **Q3_K_M** | 760MB | Lower | Very Fast | Limited use, speed-critical |

## Testing the Model

After downloading, test the integration:

```bash
# Run a quick test
python -c "from src.llm_engine import LLMEngine; llm = LLMEngine(); print(llm.analyze('Analyze AAPL stock fundamentals'))"
```

## Model Capabilities

**What DeepSeek-R1 Excels At:**
- Financial statement analysis
- Market sentiment interpretation
- Investment thesis formulation
- Risk assessment reasoning
- Multi-step analytical workflows

**Example Prompts:**
- "Analyze Tesla's Q4 earnings and provide investment recommendation"
- "Compare risk profiles of AAPL vs SPY using VaR and Sharpe ratio"
- "Evaluate the impact of Fed rate changes on tech sector stocks"

## Performance Tips

1. **GPU Acceleration**: Model automatically attempts GPU offload for faster inference
2. **Context Window**: Supports up to 131K tokens (16K default for safety)
3. **Threading**: Automatically uses 75% of CPU cores for optimal performance
4. **Quantization Trade-off**: Q6_K recommended for accuracy; Q4_K_M for speed

## Troubleshooting

**Model Not Found:**
```
Error: Model not found at models/DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf
```
→ Ensure file is in `models/` directory with exact filename

**Out of Memory:**
→ Try a smaller quantization (Q4_K_M or Q4_0)
→ Reduce `n_ctx` in `inference_script.py` from 16384 to 8192

**Slow Inference:**
→ Ensure GPU is detected (`n_gpu_layers=-1` for max GPU)
→ Use Q4_0 quantization for fastest speed
→ Check CPU thread count in logs

## Comparison: DeepSeek-R1 vs Granite 4.0

| Aspect | DeepSeek-R1-Distill-Qwen-1.5B | Granite 4.0 |
|--------|-------------------------------|-------------|
| **Size** | 1.5B params | 7B+ params |
| **Speed** | Faster (smaller model) | Slower |
| **Reasoning** | CoT, RL-enhanced | Standard |
| **Financial Focus** | SEC filings, stock analysis | General purpose |
| **Context** | 131K tokens | 16K tokens |
| **VRAM** | ~2GB (Q6_K) | ~5GB+ |

## Additional Resources

- [DeepSeek-R1 Official Repo](https://github.com/deepseek-ai/DeepSeek-R1)
- [Qwen2.5 Documentation](https://qwen.ai/)
- [llama.cpp GGUF Guide](https://github.com/ggerganov/llama.cpp)
