from ufo.utils import OpenAIChat
import os, yaml, rich, pdb
from typing import List, Union, Tuple
class UFOVerifier:
    def __init__(self, config):
        self.gpt_model = config['openai_model']
        self.gpt_baseurl = config['openai_baseurl']
        self.gpt_apikey = config['openai_apikey']
        self.gpt = OpenAIChat(
            model_name = self.gpt_model, 
            base_url = self.gpt_baseurl, 
            api_key = self.gpt_apikey)
        
        self.verification_prompt = yaml.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'config/ufo.yaml',
                ),
                'r',
            ),
            yaml.FullLoader,
        )
    
    def flatten(self, sample_claim_with_answers:List[List[Tuple]]=None):
        flattened_claims, flattened_queries, flattened_evidences, flattened_answers = [], [], [], []
        shape = []
        for claim, query, evidences, answers in sample_claim_with_answers:
            assert len(evidences) == len(answers), f'{len(evidences)}, {len(answers)}'
            shape.append(len(evidences))
            for evidence, answer in zip(evidences, answers):    
                flattened_claims.append(claim)
                flattened_queries.append(query)
                flattened_evidences.append(evidence)
                flattened_answers.append(answer)
        return flattened_claims, flattened_queries, flattened_evidences, flattened_answers, shape
    
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
                raise ValueError(f"列表A的长度 {len(list_a)} 不足以按照B的形状 {shape_b} 分割")
            result.append(list_a[index:index+length])
            index += length
        return result
    
    def __call__(self, sample_claim_with_answers:List[List[Tuple]]=None):
        # [(c1, q1, e1, a1), (c2, q2, e2, a2), ...]
        #* 给定c与a, 验证是否事实一致
        assert isinstance(sample_claim_with_answers, List), f'claim_with_answers should be a list of tuple, but got {type(sample_claim_with_answers)}'
        results, results_ppl = [], []
        #* query, claim, evidence的长度相同且一一对应, 但每个query可能对应多个evidence
        flattened_claims, flattened_queries, flattened_evidences, flattened_answers, shape = self.flatten(sample_claim_with_answers)
        #* 拆分成[(c1, q1, e11, a11), (c1, q1, e12, a12), (c2, q2, e21, a21), (c2, q2, e22, a22), ...]的形式, 一同验证后再把结果拼回原来的形状
        # rich.print(f'queries: {flattened_queries}')
        # rich.print(f'claims: {flattened_claims}')
        # rich.print(f'evidences: {flattened_evidences}')
        # rich.print(f'answers: {flattened_answers}')
        # rich.print(f'queries shape: {len(flattened_queries)}; claims shape: {len(flattened_claims)}; evidence shape: {len(flattened_evidences)}; answer shape: {len(flattened_answers)}')
        flattened_results, flattened_results_ppl = self._verification(flattened_queries, flattened_claims, flattened_evidences, flattened_answers)
        results = self.split_list_by_shape(flattened_results, shape)
        results_ppl = self.split_list_by_shape(flattened_results_ppl, shape)
        
        for _id1, (result_list, result_ppl_list) in enumerate(zip(results, results_ppl)):
            for _id2, (result, result_ppl) in enumerate(zip(result_list, result_ppl_list)):
                if result_ppl == None:  # error
                    results[_id1][_id2] = {
                        'reasoning': 'No reasoning.',
                        'factuality': False,
                        'factuality_ppl': 1e6
                    }
                else:
                    results[_id1][_id2]['factuality_ppl'] = result_ppl
        return results
    
    def _verification(self, query, claim, evidence, answer):
        '''
        claims_with_evidences每次检测一个sample
        Factool用claim和evidence对照判断
        '''
        messages_list = [
            [
                {"role": "system", "content": self.verification_prompt['system']},
                {"role": "user", "content": self.verification_prompt['user'].format(
                    claim=c, 
                    answer=a['answer'])},
            ]
            for q, c, e, a in zip(query, claim, evidence, answer)
        ]
        return self.gpt.run(messages_list, dict)