import os, yaml, tqdm, rich, time
from typing import List, Union, Tuple

class TemplateRetriever:
    def __init__(self, config):
        self.search_num = config['retriever']['template_evidence']['index_name']
        
    def __call__(self, input_samples_query:List[List[Union[List[str], str]]]=None, **kwargs) -> List[List[str]]:
        #TODO: Customize your own fact source!
        results = []
        for claims_query in tqdm.tqdm(input_samples_query):
            # obtain fact passages for each claim in an evaluated text
            result_claims = [
                ['This is a sample passage'] * self.search_num for i in range(len(claims_query))
            ]
            results.append(result_claims)
        return results