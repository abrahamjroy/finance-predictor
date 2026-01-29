pip install -U "huggingface_hub[cli]"

huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF \
  DeepSeek-R1-Distill-Qwen-1.5B-Q6_K.gguf \
  --local-dir models \
  --local-dir-use-symlinks False