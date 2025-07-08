from ufo.config import Config
from ufo.utils import get_dataset
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import Counter
import argparse, json, rich
import numpy as np
from typing import List
result_dir = '/fs/fast/u2022000150/project/ufo/output'
parser = argparse.ArgumentParser()
parser.add_argument('--save_note', type=str, default='2025_06_05_08_35_ChatGPT_ufo')
parser.add_argument('--fact_sources', type=str, nargs='+', default=['web', 'knowledge', 'human'])
parser.add_argument('--eval_methods', type=str, nargs='+')
parser.add_argument('--dataset_fn', type=str)
args = parser.parse_args()
def get_correlation(preds: List, labels: List):
    assert len(preds) == len(labels), f'length should be equal: {len(preds)}; {len(labels)}'
    p_stat = pearsonr(labels, preds)
    s_stat = spearmanr(labels, preds)
    k_stat = kendalltau(labels, preds)
    result = {
        'pearson': (float(round(p_stat.statistic, 3)), float(round(p_stat.pvalue, 3))),
        'spearmanr': (float(round(s_stat.statistic, 3)), float(round(s_stat.pvalue, 3))),
        'kendalltau': (float(round(k_stat.statistic, 3)), float(round(k_stat.pvalue, 3))),
    }
    return result

# def get_single_source_ppl(data=None, source_name=None) -> List:
#     '''只评估单个源, 根据ppl升序排列后的第一个判断结果'''
#     detail_name = f'{source_name}_details'
#     preds = []
#     for sample in data:
#         sample_score = []
#         for claim_result in sample['output']:    
#             details = claim_result[detail_name]
#             details = [(detail['factuality'], detail['factuality_ppl'], detail['reasoning']) for detail in details]
#             details.sort(key=lambda x: x[1])    # 根据ppl升序排列
#             claim_factuality = 0
#             for factuality, factuality_ppl, reasoning in details:
#                 if 'NOANS' in reasoning.upper(): # invalid evidence
#                     continue
#                 if isinstance(factuality, str):
#                     if 'TRUE' in factuality.upper():
#                         factuality = 1
#                     else:
#                         factuality = 0
#                 elif not isinstance(factuality, bool): 
#                     rich.print(f'find invalid type: {type(factuality)}; {factuality}')
#                     factuality = 0
#                 claim_factuality = factuality
#                 break
#             sample_score.append(claim_factuality)
#         if len(sample_score) == 0:
#             preds.append(0)
#         else:
#             preds.append(np.mean(sample_score))
#     return preds

# def get_single_source_majority(data=None, source_name=None) -> List:
#     '''只评估单个源, 根据ppl升序排列后的第一个判断结果'''
#     detail_name = f'{source_name}_details'
#     preds = []
#     for sample in data:
#         sample_score = []
#         for claim_result in sample['output']:    
#             details = claim_result[detail_name]
#             details = [(detail['factuality'], detail['factuality_ppl'], detail['reasoning']) for detail in details]
#             details.sort(key=lambda x: x[1])
#             #* majority
#             claim_factuality = []
#             for factuality, factuality_ppl, reasoning in details:
#                 # invalid evidence
#                 if 'NOANS' in reasoning.upper():
#                     continue
#                 if isinstance(factuality, str):
#                     if 'TRUE' in factuality.upper():
#                         factuality = 1
#                     else:
#                         factuality = 0
#                 elif not isinstance(factuality, bool): 
#                     rich.print(f'find invalid type: {type(factuality)}; {factuality}')
#                     factuality = 0
#                 claim_factuality.append(factuality)
#             if len(claim_factuality) == 0:
#                 claim_factuality = 0
#             else:
#                 mode, count = Counter(claim_factuality).most_common(1)[0]
#                 claim_factuality = mode
#             sample_score.append(claim_factuality)
#         if len(sample_score) == 0:
#             preds.append(0)
#         else:
#             preds.append(np.mean(sample_score))
#     return preds


# def get_multi_source_ppl(data=None, source_names=None):
#     detail_names = [f'{source_name}_details' for source_name in source_names]
#     preds = []
#     for sample in data:
#         sample_score = []
#         for claim_result in sample['output']:    
#             details = []
#             for detail_name in detail_names:
#                 details.extend(claim_result[detail_name])
#             details = [(detail['factuality'], detail['factuality_ppl'], detail['reasoning']) for detail in details]
#             details.sort(key=lambda x: x[1])
#             #* PPL
#             claim_factuality = False
#             for factuality, factuality_ppl, reasoning in details:
#                 # invalid evidence
#                 if 'NOANS' in reasoning.upper():
#                     continue
#                 if isinstance(factuality, str):
#                     if 'TRUE' in factuality.upper():
#                         factuality = True
#                     else:
#                         factuality = False
#                 elif not isinstance(factuality, bool): 
#                     rich.print(f'find invalid type: {type(factuality)}; {factuality}')
#                     factuality = False
#                 claim_factuality = factuality
#                 break
#             sample_score.append(claim_factuality)
#         if len(sample_score) == 0:
#             preds.append(0)
#         else:
#             try:
#                 preds.append(np.mean(sample_score))
#             except Exception as e:
#                 print(sample_score)
#                 raise
#     return preds

def get_multi_source_majority(data=None, unordered_source_names=None):
    '''不考虑次序, 从所有source中先筛掉noans, 剩下的当中挑true, false中的大多数'''
    source_names = [f'{source_name}_sources' for source_name in unordered_source_names]
    extraction_names = [f'{source_name}_extractions' for source_name in unordered_source_names]
    detail_names = [f'{source_name}_details' for source_name in unordered_source_names]
    preds, from_sources = [], []
    for sample in data:
        sample_score, sample_from_sources = [], []
        # 验证该sample下的所有claim
        for claim_result in sample['output']:    
            claim, queries = claim_result['claims'], claim_result['queries']
            sources, extractions, details, cur_detail_name = [], [], [], []
            #* load evidence, extracted answers, details
            for source_name, extraction_name, detail_name in zip(source_names, extraction_names, detail_names):
                sources.extend(claim_result[source_name])
                extractions.extend(claim_result[extraction_name])
                details.extend(claim_result[detail_name])
                cur_detail_name.extend([detail_name] * len(claim_result[detail_name]))
            #& verify sources together
            claim_factuality, claim_judgment = [], []
            #* majority
            for source_id, (_source, _extraction, _detail, _detail_name) in enumerate(zip(sources, extractions, details, cur_detail_name)):
                evidence = _source['evidence']
                ext_reasoning, answer = _extraction['reasoning'], _extraction['answer']
                factuality, factuality_ppl, factuality_reasoning = _detail['factuality'], _detail['factuality_ppl'], _detail['reasoning']
                #* invalid answer
                if 'NOANS' in ext_reasoning.upper() or 'NOANS' in str(answer).upper() or 'NOANS' in factuality_reasoning.upper(): continue
                if isinstance(factuality, str):
                    if 'TRUE' in factuality.upper():
                        factuality = 1
                    else:
                        factuality = 0
                elif isinstance(factuality, bool):
                    factuality = int(factuality)
                elif not isinstance(factuality, bool): 
                    rich.print(f'find invalid type: {type(factuality)}; {factuality}')
                    factuality = 0
                claim_factuality.append(factuality)
                claim_judgment.append({
                    'claim': claim,
                    "factuality": factuality,
                    'query': ' '.join(queries),
                    'source': _detail_name.replace('_details', ''),
                    'evidence': evidence,
                    'answer': str(answer),
                    'reasoning': factuality_reasoning
                })
            if len(claim_factuality) == 0:
                #* 所有事实源全是NOANS
                sample_score.append(0)
                sample_from_sources.append({
                    'claim': claim,
                    'factuality': 0,
                    'query': ' '.join(queries),
                    'source': 'NOANS',
                    'evidence': 'NOANS',
                    'answer': 'NOANS',
                    'reasoning': 'NOANS',
                })
            else:
                #* 取众数
                mode, count = Counter(claim_factuality).most_common(1)[0]
                sample_score.append(mode)
                is_append = False
                for _judgment in claim_judgment:
                    if _judgment['factuality'] == mode and 'NOANS' not in _judgment['answer'].upper() and 'NOANS' not in _judgment['reasoning'].upper():
                        sample_from_sources.append(_judgment)
                        is_append = True
                        break
                assert is_append == True
        if len(sample_score) == 0:
            #* claim 数量为0
            preds.append(0)
            from_sources.append([])
        else:
            preds.append(np.mean(sample_score))
            from_sources.append(sample_from_sources)
    return preds, from_sources

# def get_multi_source_seq(data=None, ordered_source_names=['web', 'knowledge', 'human']):
#     detail_names = [f'{source_name}_details' for source_name in ordered_source_names]
#     preds = []
#     for sample in data:
#         sample_score = []
#         for claim_result in sample['output']:    
#             details = []
#             for detail_name in detail_names:
#                 details.append(claim_result[detail_name])
#             #& verify with source one-by-one
#             claim_factuality = 0
#             for source_detail in details:
#                 source_detail = [(detail['factuality'], detail['factuality_ppl'], detail['reasoning']) for detail in source_detail]
#                 source_detail.sort(key=lambda x: x[1])
#                 #& cannot find an answer from all evidences in such source
#                 for factuality, factuality_ppl, reasoning in source_detail:
#                     # invalid evidence, go find the next
#                     if 'NOANS' in reasoning.upper():
#                         continue
#                     if isinstance(factuality, str):
#                         if 'TRUE' in factuality.upper():
#                             factuality = 1
#                         else:
#                             factuality = 0
#                     elif isinstance(factuality, bool):
#                         factuality = int(factuality)
#                     elif not isinstance(factuality, bool): 
#                         rich.print(f'find invalid type: {type(factuality)}; {factuality}')
#                         factuality = 0
#                     claim_factuality = factuality
#                     break
#             sample_score.append(claim_factuality)
#         if len(sample_score) == 0:
#             preds.append(0)
#         else:
#             preds.append(np.mean(sample_score))
#     return preds










def get_multi_source_seq_majority(data=None, ordered_source_names=['web', 'knowledge', 'human']):
    '''按给定事实源顺序, 每个事实源内采用多数投票, 除非全是noans'''
    source_names = [f'{source_name}_sources' for source_name in ordered_source_names]
    extraction_names = [f'{source_name}_extractions' for source_name in ordered_source_names]
    detail_names = [f'{source_name}_details' for source_name in ordered_source_names]
    preds = []
    from_sources = []
    for sample in data:
        sample_score = []
        sample_from_sources = []
        for claim_result in sample['output']:
            claim, queries = claim_result['claims'], claim_result['queries']
            sources, extractions, details = [], [], []
            #* load evidence, extracted answers, details
            for source_name, extraction_name, detail_name in zip(source_names, extraction_names, detail_names):
                sources.append(claim_result[source_name])
                extractions.append(claim_result[extraction_name])
                details.append(claim_result[detail_name])
            #& verify with source one-by-one
            for source_id, (_source, _extraction, _detail) in enumerate(zip(sources, extractions, details)):
                #* verify in a single fact source
                _source = [source['evidence'] for source in _source]
                _extraction = [(extraction['reasoning'], extraction['answer']) for extraction in _extraction]
                _detail = [(detail['factuality'], detail['factuality_ppl'], detail['reasoning']) for detail in _detail]
                # source_detail.sort(key=lambda x: x[1])
                #* majority
                claim_factuality, claim_judgment = [], []
                for (evidence), (ext_reasoning, answer), (factuality, factuality_ppl, reasoning) in zip(_source, _extraction, _detail):
                    # invalid evidence, move to the next source
                    if 'NOANS' in reasoning.upper() or 'NOANS' in ext_reasoning.upper() or 'NOANS' in str(answer).upper():
                        continue
                    if isinstance(factuality, str):
                        if 'TRUE' in factuality.upper():
                            factuality = 1
                        else:
                            factuality = 0
                    elif isinstance(factuality, bool):
                        factuality = int(factuality)
                    elif not isinstance(factuality, bool): 
                        rich.print(f'find invalid type: {type(factuality)}; {factuality}')
                        factuality = 0
                    claim_factuality.append(factuality)
                    claim_judgment.append({
                        'claim': claim,
                        "factuality": factuality,
                        'query': ' '.join(queries),
                        'source': detail_names[source_id].replace('_details', ''),
                        'evidence': evidence,
                        'answer': str(answer),
                        'reasoning': reasoning
                    })
                if len(claim_factuality) == 0 and source_id < len(sources) - 1:
                    #* 该事实源全是NOANS, 去下一个事实源
                    continue
                elif len(claim_factuality) == 0 and source_id == len(sources) - 1:
                    #* 该事实源全是NOANS，且没下一个事实源
                    sample_score.append(0)
                    sample_from_sources.append({
                        'claim': claim,
                        'factuality': 0,
                        'query': ' '.join(queries),
                        'source': 'NOANS',
                        'evidence': 'NOANS',
                        'answer': 'NOANS',
                        'reasoning': 'NOANS',
                    })
                else:
                    #* len(claim_factuality) > 0: 取众数, 然后退出
                    mode, count = Counter(claim_factuality).most_common(1)[0]
                    sample_score.append(mode)
                    is_append = False
                    for _judgment in claim_judgment:
                        if _judgment['factuality'] == mode and 'NOANS' not in _judgment['answer'].upper() and 'NOANS' not in _judgment['reasoning'].upper():
                            sample_from_sources.append(_judgment)
                            is_append = True
                            break
                    assert is_append == True
                    break
        if len(sample_score) == 0:
            # 该事实没有任何验证结果, 返回0
            preds.append(0)
            from_sources.append([])
        else:
            preds.append(np.mean(sample_score))
            from_sources.append(sample_from_sources)
    return preds, from_sources
                    
                    
                    
                    
if __name__ == "__main__":
    path = f'{result_dir}/{args.save_note}/{args.dataset_fn}.json'
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    labels = [line['label'] for line in data]
    try:
        rich.print(f'average label:', round(np.mean(labels), 3))
        is_label = 1
    except Exception as e:
        rich.print(f'No ground truth labels.')
        is_label = 0
        
    eval_method_dict = {
        # 'single_ppl': get_single_source_ppl,
        # 'single_major': get_single_source_majority,
        # 'multi_sequence': get_multi_source_seq,
        'multi_sequence_major': get_multi_source_seq_majority,
        # 'multi_ppl': get_multi_source_ppl,
        'multi_major': get_multi_source_majority,
    }
    for cur_eval_method_name in args.eval_methods:
        eval_method = eval_method_dict[cur_eval_method_name]
        rich.print(f'*** eval method: {cur_eval_method_name} ***')
        # if 'single' in cur_eval_method_name:
        #     for fact_source in args.fact_sources:
        #         preds = eval_method(data=data, source_name=fact_source)
        #         avg_preds = round(np.mean(preds), 3)
        #         rich.print(f'{fact_source} {len(preds)} preds avg: {avg_preds}')
        #         rich.print(f'{fact_source} {len(preds)} preds: {preds}')
        #         if is_label:
        #             rich.print("*"*30)
        #             rich.print(f'{fact_source} correlations: ')
        #             rich.print(get_correlation(preds, labels))
        if cur_eval_method_name == 'multi_sequence' or cur_eval_method_name == 'multi_sequence_major':
            fact_sources = [
                ['human'],
                ['web'],
                ['knowledge'],
                ['human', 'web', 'knowledge'],
                # ['human', 'knowledge', 'web'],
                # ['web', 'knowledge', 'human'],
                # ['web', 'human', 'knowledge'],
                # ['knowledge', 'human', 'web'],
                # ['knowledge', 'web', 'human'],
            ]
            for fact_source in fact_sources:
                rich.print("*"*20)
                rich.print(f'evaluate with source: {fact_source}')
                preds, from_sources = eval_method(data=data, ordered_source_names=fact_source)
                # rich.print(f'from sources: {from_sources[0]}')
                source2count = {item: [0, 0] for item in fact_source}   # False, True
                source2count['NOANS'] = [0, 0]
                for _source in from_sources:
                    for item in _source:
                        if item['factuality'] == 1:
                            source2count[item['source']][1] += 1
                        else:
                            source2count[item['source']][0] += 1
                rich.print(f'source2count: {source2count}')
                rich.print(f'evaluation scores: {np.mean(preds):.3f}')
                if is_label:
                    rich.print(f'{fact_source} correlations: ')
                    rich.print(get_correlation(preds, labels))
        elif cur_eval_method_name == 'multi_major': 
            fact_sources = [
                ['human'],
                ['web'],
                ['knowledge'],
                ['human', 'web', 'knowledge'],
                # ['human', 'knowledge', 'web'],
                # ['web', 'knowledge', 'human'],
                # ['web', 'human', 'knowledge'],
                # ['knowledge', 'human', 'web'],
                # ['knowledge', 'web', 'human'],
            ]
            for fact_source in fact_sources:
                rich.print("*"*20)
                rich.print(f'evaluate with source: {fact_source}')
                preds, from_sources = eval_method(data=data, unordered_source_names=fact_source)
                source2count = {item: [0, 0] for item in fact_source}   # False, True
                source2count['NOANS'] = [0, 0]
                for _source in from_sources:
                    for item in _source:
                        if item['factuality'] == 1:
                            source2count[item['source']][1] += 1
                        else:
                            source2count[item['source']][0] += 1
                rich.print(f'source2count: {source2count}')
                rich.print(f'evaluation scores: {np.mean(preds):.3f}')
                if is_label:
                    rich.print(f'multi_major correlations: ')
                    rich.print(get_correlation(preds, labels))
        else: raise NotImplementedError