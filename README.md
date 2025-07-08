# 🛸 UFO: A Unified Framework for Factuality Evaluation with Multiple Plug-and-Play Fact Sources

Submission to EMNLP 2025 (System Demonstration Track)

UFO is a unified and extensible framework designed for evaluating the factuality of LLM outputs, leveraging multiple plug-and-play fact sources including human-written evidence, web search results, and LLM knowledge.

## 🛠️ Getting Started

### 1. Install Dependencies
```shell
pip install -r requirements.txt
```

### 2. Configure the System
Update the following values in `config.yaml`:
- `openai_apikey`
- `openai_baseurl`
- `openai_model`

You can use either online LLM inference services or local backends:
- `Online Services`: [OpenAI](https://openai.com) and Siliconflow [Siliconflow](https://www.siliconflow.cn)
- `Local Backend` (via vLLM): 
```shell
CUDA_VISIBLE_DEVICES=0 nohup vllm serve <model_name> --dtype auto --api-key EMPTY --port 2233 --tensor-parallel-size 1 > qwen2.5-14b.log 2>&1 &
```
✅ Default model: `Qwen2.5-14B-Instruct`


## ⚙️ Configuration Overview

## 🚀 Run Evaluation with UFO
Run the evaluation pipeline:
```bash
bash run_exp.sh
```

#TODO
### Usage
Please set the default settings in `config.yaml`, including:
1. output settings: `save_dir` and `dataset_name`.
2. inference setting: `openai_apikey`, `openai_baseurl`, and `openai_model`.
3. fact source setting: set the config in `config.yaml`



To directly use:
```
bash run_exp.sh
```
Please set the value of DATASET_NAME, SUB_NAME, OPENAI_APIKEY, OPENAI_BASEURL, and OPENAI_MODEL.



## 🌐 Interactive Web Demo
Launch the interactive demo interface:
```shell
python app.py
```
Access the web UI in your browser and evaluate using `example_data_upload.jsonl`, or any JSONL files with each line containing two keys: `response` for the evaluated LLM text, and `reference_answers` for any relevant reference (optional).




## 📄 License
```sql
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```