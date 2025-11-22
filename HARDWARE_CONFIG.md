# Hardware Configuration Guide

## Automatic Hardware Optimization

The Finance Predictor application automatically detects and optimizes for your hardware configuration. No manual configuration is required!

## Supported Hardware

### NVIDIA GPUs
- **Automatic Detection**: The application automatically detects NVIDIA GPUs via CUDA
- **Supported Cards**: Any NVIDIA GPU with CUDA support (GTX 1060+, RTX series, Tesla, etc.)
- **Optimization**: All model layers are loaded to GPU memory for maximum performance
- **Fallback**: Automatically falls back to CPU if GPU is unavailable

### AMD CPUs
- **Thread Optimization**: Automatically uses 75% of available CPU cores
- **Ideal For**: AMD Ryzen, Threadripper, and EPYC processors
- **Multi-Core Scaling**: Better performance with more physical cores

### Intel CPUs
- **Compatible**: Fully compatible with Intel Core and Xeon processors
- **Performance**: Same automatic thread optimization as AMD

## Component-Specific Optimizations

### LLM Engine (Phi-4 Mini Reasoning)
- **Auto-Detection**: Checks for NVIDIA GPU via PyTorch/CUDA
- **GPU Mode**: Loads all model layers to VRAM (`n_gpu_layers=-1`)
- **CPU Mode**: Falls back to optimized CPU inference if no GPU detected
- **Thread Count**: Dynamically calculated based on available CPU cores

### XGBoost Forecasting
- **CUDA Acceleration**: Attempts GPU acceleration first (`device="cuda"`)
- **CPU Fallback**: Automatically switches to multi-threaded CPU if GPU unavailable
- **Tree Method**: Uses histogram-based algorithm for efficiency

## Performance Tips

### For GPU Users
1. Ensure NVIDIA drivers are up to date
2. Install CUDA toolkit (11.7+ recommended)
3. Verify llama-cpp-python was built with CUDA support:
   ```bash
   pip uninstall llama-cpp-python
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
   ```

### For CPU-Only Users
1. The application works perfectly on CPU
2. Inference will be slower but still functional
3. Consider upgrading to a CUDA-capable GPU for 5-10x speedup

### For AMD Ryzen/Threadripper Users
1. The application optimizes for physical core count
2. SMT/Hyper-threading is automatically accounted for
3. Higher core counts will see better XGBoost performance

## Checking Your Configuration

The application logs its hardware detection on startup. Look for messages like:

```
Detected NVIDIA GPU: NVIDIA GeForce RTX 4070
GPU Memory: 12.0 GB
Detected 16 CPU cores, using 12 threads
✓ Model loaded with GPU acceleration (-1 layers)
```

Or for CPU-only systems:

```
No NVIDIA GPU detected, using CPU-only mode
Detected 8 CPU cores, using 6 threads
✓ Model loaded in CPU-only mode
```

## Troubleshooting

### GPU Not Detected
- Verify NVIDIA drivers are installed
- Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall llama-cpp-python with CUDA support (see above)

### Poor Performance
- Check if models are loading correctly
- Monitor GPU usage (Task Manager on Windows, `nvidia-smi` on Linux)
- Ensure you're using the Q4_K_M quantized model (2.3GB, not full precision)

### Out of Memory
- The Phi-4 Q4_K_M model requires ~2.5GB VRAM
- Close other GPU-intensive applications
- Consider using a smaller model or CPU-only mode
