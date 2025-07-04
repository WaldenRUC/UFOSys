## UFO: A Unified Framework for Factuality Evaluation with Multiple Plug-and-Play Fact Sources

Submission to EMNLP 2025 system demonstration track.

### Start the LLM evaluator
The UFO system supports evaluation with local vLLM backend. You can start a vLLM inference server as:

```
CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve Qwen2.5-14B-Instruct --dtype auto --api-key EMPTY --port 2233 --tensor-parallel-size 2 > qwen2.5-14b.log 2>&1 &
```

