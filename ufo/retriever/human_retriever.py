from elasticsearch import Elasticsearch
import os, yaml, tqdm, rich
from typing import List, Union
from ufo.retriever.elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
class HumanRetriever:
    def __init__(self, config):
        self.index_name = config['retriever']['human_written_evidence']['index_name']
        self.search_num = config['retriever']['human_written_evidence']['search_num']
        self.es = ElasticSearchBM25Retriever.create(config['retriever']['human_written_evidence']['es_url'], self.index_name)
        print(f'loaded index: {self.index_name}, search num: {self.search_num}, size: {self.es.get_document_count()}')

    def __call__(self, input_samples_query:List[List[Union[List[str], str]]]=None, human_answers=None):
        if human_answers == None:
            human_answers = [''] * len(input_samples_query)
        assert len(input_samples_query) == len(human_answers)
        results = []
        for claims_query, sample_answer in tqdm.tqdm(list(zip(input_samples_query, human_answers)), desc='Retrieving human evid', ncols=100):
            result_claims = []
            for claim_query in claims_query:    
                evidences = self.searching(claim_query, search_num=self.search_num)
                for answer in sample_answer:
                    evidences.append({
                        'evidence': answer,
                        'score': 0
                    })
                result_claims.append(evidences)
            # result_claims: [[evid1, score1], [evid2, score2], ...]
            results.append(result_claims)
        return results
        
    
    def searching(self, query:str=None, search_num:int=3):
        if query is None: return [] # catch error
        if isinstance(query, list): # specified for factool
            query = ' '.join(query)
        results = []
        assert isinstance(query, str), query
        # 执行查询
        docs_and_scores = self.es.get_relevant_documents(query, num_docs=search_num)
        for doc, score in docs_and_scores:
            evidence = doc.page_content
            results.append({
                'evidence': evidence,
                'score': score,
            })
        return results

                
if __name__ == '__main__':
    import sys, rich
    sys.path.append('/data00/zhaoheng_huang/project/ufo-emnlp/demo')
    es = Elasticsearch("http://localhost:9200", timeout=18000)
    retriever = HumanRetriever({'es':es, 'index_name':'bm25_psgs_index'})
    rich.print(retriever('Joe Biden'))