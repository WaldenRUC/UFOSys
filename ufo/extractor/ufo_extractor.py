from ufo.utils import OpenAIChat
import os, yaml, rich
from typing import List, Union, Tuple
class UFOExtractor:
    def __init__(self, config):
        self.gpt_model = config['openai_model']
        self.gpt_baseurl = config['openai_baseurl']
        self.gpt_apikey = config['openai_apikey']
        self.gpt = OpenAIChat(
            model_name = self.gpt_model, 
            base_url = self.gpt_baseurl, 
            api_key = self.gpt_apikey)
        self.extraction_prompt = yaml.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'config/ufo.yaml',
                ),
                'r',
            ),
            yaml.FullLoader,
        )
        
    def flatten(self, sample_claim_with_evidences:List[List[Tuple]]=None):
        flattened_claims, flattened_queries, flattened_evidences = [], [], []
        shape = []
        for claim, query, evidences in sample_claim_with_evidences:
            shape.append(len(evidences))
            for evidence in evidences:    
                flattened_claims.append(claim)
                flattened_queries.append(query)
                flattened_evidences.append(evidence)
        return flattened_claims, flattened_queries, flattened_evidences, shape
    
    def split_list_by_shape(self, list_a, shape_b):
        """
        将列表A按照列表B的形状分割
        
        参数:
            list_a: 要分割的列表
            shape_b: 包含整数的列表，指定每个子列表的长度
            
        返回:
            分割后的列表的列表
        """
        result = []
        index = 0
        for length in shape_b:
            if index + length > len(list_a):
                raise ValueError("列表A的长度不足以按照B的形状分割")
            result.append(list_a[index:index+length])
            index += length
        return result
    
    def __call__(self, sample_claim_with_evidences:List[List[Tuple]]=None):
        # [(c1, q1, e1), (c2, q2, e2), ...]
        assert isinstance(sample_claim_with_evidences, List), f'claims_with_evidences should be a list of tuple, but got {type(sample_claim_with_evidences)}'
        results, results_ppl = [], []
        #* query, claim, evidence的长度相同且一一对应, 但每个query可能对应多个evidence
        flattened_claims, flattened_queries, flattened_evidences, shape = self.flatten(sample_claim_with_evidences)
        #* 拆分成[(c1, q1, e11), (c1, q1, e12), (c2, q2, e21), (c2, q2, e22), ...]的形式, 一同抽取答案后再把结果拼回原来的形状
        flattened_results, flattened_results_ppl = self._extraction(flattened_queries, flattened_claims, flattened_evidences)
        results = self.split_list_by_shape(flattened_results, shape)
        results_ppl = self.split_list_by_shape(flattened_results_ppl, shape)
        for _id1, (result_list, result_ppl_list) in enumerate(zip(results, results_ppl)):
            for _id2, (result, result_ppl) in enumerate(zip(result_list, result_ppl_list)):
                if result_ppl == None:  # error
                    results[_id1][_id2] = {
                        'reasoning': 'No answer.',
                        'answer': 'NOANS',
                        'answer_ppl': 1e6
                    }
                else:
                    results[_id1][_id2]['answer_ppl'] = result_ppl
        return results
        
    def _extraction(self, query, claim, evidence):
        '''
        claims_with_evidences每次检测一个sample
        Factool用claim和evidence对照判断
        '''
        formatted_query = []
        for _q in query:
            if isinstance(_q, str):
                formatted_query.append(_q)
            elif isinstance(_q, list):
                formatted_query.append(' '.join(_q))
            else:
                formatted_query.append('No query.')
        query = formatted_query
        # query = [_q if isinstance(_q, str) else ' '.join(_q) for _q in query]
        messages_list = [
            [
                {"role": "system", "content": self.extraction_prompt['system']},
                {"role": "user", "content": self.extraction_prompt['user'].format(
                    question=q, 
                    evidence=e)},
            ]
            for q, c, e in zip(query, claim, evidence)
        ]
        return self.gpt.run(messages_list, dict)