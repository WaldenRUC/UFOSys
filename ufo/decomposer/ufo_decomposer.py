from ufo.utils import OpenAIChat
import os, yaml, rich
from typing import List, Union

class UFODecomposer:
    def __init__(self, config):
        self.gpt_model = config['openai_model']
        self.gpt_baseurl = config['openai_baseurl']
        self.gpt_apikey = config['openai_apikey']
        self.gpt = OpenAIChat(
            model_name = self.gpt_model, 
            base_url = self.gpt_baseurl, 
            api_key = self.gpt_apikey)
        
        self.claim_prompt = yaml.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'config/ufo.yaml',
                ),
                'r',
            ),
            yaml.FullLoader,
        )
        
        print(f'UFODecomposer initiated!')
    
    def __call__(self, input_text:Union[List[str], str]=None):
        if isinstance(input_text, str):
            input_texts = [input_text]
        elif isinstance(input_text, list):
            input_texts = input_text
        while True:
            try:    
                claims, claims_ppl = self._claim_extraction(responses=input_texts)
                extracted_claims = [
                    [claim["claim"] for claim in claim_list] 
                    for claim_list in claims]
                break
            except Exception as e:
                print(f'error <{e}> in decomposer. input texts: <{input_texts}>. retry...')
        return extracted_claims, claims_ppl
    
    def _claim_extraction(self, responses):
        messages_list = [
            [
                {"role": "system", "content": self.claim_prompt["system"]},
                {
                    "role": "user",
                    "content": self.claim_prompt["user"].format(input=response),
                },
            ]
            for response in responses
        ]
        return self.gpt.run(messages_list, List)