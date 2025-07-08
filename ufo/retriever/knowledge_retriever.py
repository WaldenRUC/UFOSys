from ufo.utils import OpenAIChat
import os, yaml, tqdm, rich, time
from typing import List, Union, Tuple
class KnowledgeRetriever:
    def __init__(self, config):
        self.gpt_model = config['openai_model']
        self.gpt_baseurl = config['openai_baseurl']
        self.gpt_apikey = config['openai_apikey']
        self.temperature = config['retriever']['llm_knowledge']['temperature']
        self.n = config['retriever']['llm_knowledge']['sampled_n']
        self.top_p = config['retriever']['llm_knowledge']['top_p']
        self.max_tokens = config['retriever']['llm_knowledge']['max_tokens']
        self.gpt = OpenAIChat(
            model_name = self.gpt_model,
            base_url = self.gpt_baseurl, 
            api_key = self.gpt_apikey,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=self.n,
            top_p=self.top_p)
        
        self.knowledge_prompt = yaml.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'config/knowledge_retriever.yaml',
                ),
                'r',
            ),
            yaml.FullLoader,
        )
    
    def __call__(self, input_samples_query:List[List[Union[List[str], str]]]=None, **kwargs):
        results = []
        for claims_query in tqdm.tqdm(input_samples_query, desc='Retrieving LLM knowledge evid', ncols=100):
            # result_claims = []
            # for claim_query in claims_query:    
            #     evidences = self.searching(claim_query)
            #     result_claims.append(evidences)
            result_claims = self.searching(claims_query)
            results.append(result_claims)
        return results
    
    def searching(self, claims_query:List[str]=None):
        if claims_query is None: return [] # catch error
        # specified for factool
        formatted_claim_query = []
        for claim_query in claims_query:
            if isinstance(claim_query, str):
                formatted_claim_query.append(claim_query)
            elif isinstance(claim_query, list):
                formatted_claim_query.append(' '.join(claim_query))
            else:
                formatted_claim_query.append(None)
        claims_query = formatted_claim_query
        results = []
        # 执行查询
        while True:
            knowledge_texts, knowledge_texts_ppl = self._gen_knowledge(claims_query)
            if None in knowledge_texts:
                rich.print(f'error in generating knowledge: {knowledge_texts}')
                time.sleep(1)
            else:
                break
        # rich.print(f'generate knowledge: {knowledge_texts}')
        for i in range(len(knowledge_texts)):
            results_sample = []
            for knowledge_text, knowledge_text_ppl in zip(knowledge_texts[i], knowledge_texts_ppl[i]):
                results_sample.append({
                    'evidence': knowledge_text,
                    'evidence_ppl': knowledge_text_ppl
                })
            results.append(results_sample)
        return results
        
        
    def _gen_knowledge(self, queries: List[str]=None):
        messages_list = [
            [
                {"role": "system", "content": self.knowledge_prompt['system']},
                {"role": "user", "content": self.knowledge_prompt['user'].format(question=query)}
            ]
            for query in queries
        ]
        
        return self.gpt.run(messages_list, str)