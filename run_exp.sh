DATASET_NAME=factool-qa
SUB_NAME=GPT4

#& ========== Local vLLM ==========
OPENAI_APIKEY=EMPTY
OPENAI_BASEURL=http://0.0.0.0:2233/v1/
OPENAI_MODEL=Qwen2.5-14B-Instruct

#& ========== OpenAI ==========
# OPENAI_APIKEY=xxx
# OPENAI_BASEURL=https://api.openai.com/v1/
# OPENAI_MODEL=gpt-4o-mini

#& ========== Siliconflow ==========
# OPENAI_APIKEY=xxx
# OPENAI_BASEURL=https://api.siliconflow.cn/v1/
# OPENAI_MODEL=Qwen/Qwen2.5-14B-Instruct


python -O run_exp.py \
    --data_dir dataset/$DATASET_NAME \
    --dataset_name $SUB_NAME \
    --method_name ufo \
    --openai_apikey $OPENAI_APIKEY \
    --openai_baseurl $OPENAI_BASEURL \
    --openai_model $OPENAI_MODEL \
    --batch_size 5