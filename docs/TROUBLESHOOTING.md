# Troubleshooting

If FAISS import fails, install `faiss-cpu`. If local model loading fails, verify paths in `configs/paths.yaml`. If LLM inference runs out of GPU memory, reduce `max_new_tokens`, use a smaller model, or enable quantization externally.
