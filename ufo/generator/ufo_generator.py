from ufo.utils import OpenAIChat
import os, yaml
from typing import List, Union

class UFOGenerator:
    def __init__(self, config):
        self.gpt_model = config['openai_model']
        self.gpt_baseurl = config['openai_baseurl']
        self.gpt_apikey = config['openai_apikey']
        self.gpt = OpenAIChat(
            model_name = self.gpt_model, 
            base_url = self.gpt_baseurl, 
            api_key = self.gpt_apikey)
        self.query_prompt = yaml.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "config/ufo.yaml",
                ),
                "r",
            ),
            yaml.FullLoader,
        )
        print(f'UFOGenerator initiated!')
        
    def __call__(self, input_claim:Union[List[str], str]=None):
        if isinstance(input_claim, str):
            input_claims = [input_claim]
        elif isinstance(input_claim, list):
            input_claims = input_claim
        while True:
            queries, queries_ppl = self._query_generation(claims=input_claims)
            if None in queries:
                print(f'find None in queries: {queries}')
                print(f'input claims: {input_claims}')
            else: 
                break
        return queries, queries_ppl
        
        
    def _query_generation(self, claims):
        messages_list = [
            [
                {"role": "system", "content": self.query_prompt["system"]},
                {
                    "role": "user",
                    "content": self.query_prompt["user"].format(input=claim),
                },
            ]
            for claim in claims
        ]
        return self.gpt.run(messages_list, List)