from __future__ import annotations
import os, importlib, yaml, openai, ast, asyncio, os, pathlib, re, rich
from typing import Any, List, Union
from openai import AsyncOpenAI
from ufo.dataset.dataset import Dataset
import numpy as np

class OpenAIChat():
    def __init__(
            self,
            model_name=None, base_url=None, api_key=None,
            max_tokens=2500, temperature=0, top_p=1,
            n=1, top_logprobs=3, request_timeout=1200,
    ):
        if model_name is None:
            model_name = os.environ.get("OPENAI_MODEL_NAME", None)
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASEURL", None)
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        assert api_key is not None, "Please set the OPENAI_API_KEY environment variable."
        assert api_key != '', "Please set the OPENAI_API_KEY environment variable."
        self.client = AsyncOpenAI(
            api_key = api_key,
            base_url = base_url,
        )
        self.config = {
            'model_name': model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'logprobs': True,
            'top_logprobs': top_logprobs,
            'n': n,
            'request_timeout': request_timeout,
            'openai_baseurl': base_url
        }

    def extract_list_from_string(self, input_string):
        # pattern = r'\[.*\]'  
        # result = re.search(pattern, input_string)
        # if result:
        #     return result.group()
        # else:
        #     return None
        start_index = input_string.find('[')
        end_index = input_string.rfind(']')

        if start_index != -1 and end_index != -1 and start_index < end_index:
            return input_string[start_index:end_index + 1]
        else:
            return None

    def extract_dict_from_string(self, input_string):
        start_index = input_string.find('{')
        end_index = input_string.rfind('}')

        if start_index != -1 and end_index != -1 and start_index < end_index:
            return input_string[start_index:end_index + 1]
        else:
            return None

    def _boolean_fix(self, output):
        return output.replace("true", "True").replace("false", "False")

    def _type_check(self, output, expected_type):
        try:
            if expected_type == str: return output
            output_eval = ast.literal_eval(output)
            if not isinstance(output_eval, expected_type):
                #* 注意这里可能返回None
                return None
            return output_eval
        except:
            '''
            if(expected_type == List):
                valid_output = self.extract_list_from_string(output)
                output_eval = ast.literal_eval(valid_output)
                if not isinstance(output_eval, expected_type):
                    return None
                return output_eval
            elif(expected_type == dict):
                valid_output = self.extract_dict_from_string(output)
                output_eval = ast.literal_eval(valid_output)
                if not isinstance(output_eval, expected_type):
                    return None
                return output_eval
            '''
            return None

    async def dispatch_openai_requests(
            self,
            messages_list,
    ) -> list[Union[List[str], str]]:
        """Dispatches requests to OpenAI API asynchronously.
        
        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        Returns:
            List of responses from OpenAI API.
        """

        async def _request_with_retry(messages, retry=3):
            for _ in range(retry):
                try:
                    if 'siliconflow' in self.config['openai_baseurl']:
                        response = await self.client.chat.completions.create(
                            model=self.config['model_name'],
                            messages=messages,
                            max_tokens=self.config['max_tokens'],
                            temperature=self.config['temperature'],
                            top_p=self.config['top_p'],
                            n=self.config['n'],
                        )    
                    else:
                        response = await self.client.chat.completions.create(
                            model=self.config['model_name'],
                            messages=messages,
                            max_tokens=self.config['max_tokens'],
                            temperature=self.config['temperature'],
                            top_p=self.config['top_p'],
                            logprobs=self.config['logprobs'],
                            top_logprobs=self.config['top_logprobs'],
                            n=self.config['n'],
                        )    
                    return response
                except openai.RateLimitError:
                    await asyncio.sleep(1)
                # except openai.Timeout:
                #     await asyncio.sleep(1)
                except openai.APIError:
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f'Unknown error: {e}')
                    await asyncio.sleep(1)
            return None

        async_responses = [
            _request_with_retry(messages)
            for messages in messages_list
        ]

        return await asyncio.gather(*async_responses)

    def get_ppl(self, logprobs: list=None):
        nll = -sum(logprobs)  # 负对数似然
        ppl = float(np.exp(nll / len(logprobs)))  # 平均后指数化
        return ppl
    
    def run(self, messages_list, expected_type):
        retry = 5
        responses = [None for _ in range(len(messages_list))]
        responses_ppl = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]
        # print(f'fact checkers solvers messages_list: {messages_list}')

        while retry > 0 and len(messages_list_cur_index) > 0:
            # print(f'fact checkers solvers {retry} retry left...')
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]

            predictions = asyncio.run(self.dispatch_openai_requests(
                messages_list=messages_list_cur,
            ))
            
            if 'siliconflow' in self.config['openai_baseurl']:
                preds = [self._type_check(
                        self._boolean_fix(prediction.choices[0].message.content),
                        expected_type) 
                    if prediction is not None else None for prediction in predictions]
                preds_ppl = [None] * len(predictions)
            else:
                #& n=1
                # if len(predictions[0].choices) == 1:
                if self.config['n'] == 1:
                    preds = [self._type_check(
                        self._boolean_fix(prediction.choices[0].message.content),
                        expected_type) 
                    if prediction is not None else None for prediction in predictions]
                    preds_ppl = [
                            self.get_ppl(
                                [content.logprob for content in prediction.choices[0].logprobs.content]
                            ) 
                        if prediction is not None else None for prediction in predictions]
                else:
                    preds = [
                        [self._type_check(
                                self._boolean_fix(choice.message.content), expected_type
                            ) for choice in prediction.choices
                        ]
                        if prediction is not None else None for prediction in predictions
                    ]
                    preds_ppl = [
                            [
                                self.get_ppl(
                                    [content.logprob for content in choice.logprobs.content]
                                )
                                for choice in prediction.choices]
                        if prediction is not None else None for prediction in predictions]
                

            finised_index = []
            for i, (pred, pred_ppl) in enumerate(zip(preds, preds_ppl)):
                # print(f'pred ppl: {pred_ppl}')
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    responses_ppl[messages_list_cur_index[i]] = pred_ppl
                    finised_index.append(messages_list_cur_index[i])

            messages_list_cur_index = [i for i in messages_list_cur_index if i not in finised_index]

            retry -= 1
            # print(f'response_ppl: {responses_ppl}')
        # print(f'fact checkers solvers responses: {responses}')
        return responses, responses_ppl
    

def get_dataset(config):
    '''load dataset from config.'''
    SUPPORT_FILES = ["jsonl", "json", "parquet"]
    dataset_path = config["dataset_path"]
    all_split = ['test']          # 仅考虑test数据集, 注意数据集命名方式
    split_dict = {split: None for split in all_split}
    for split in all_split:
        exist_flag = 0
        for file_postfix in SUPPORT_FILES:
            split_path = os.path.join(dataset_path, f'{split}.{file_postfix}')
            if not os.path.exists(split_path):
                continue
            else:
                exist_flag = 1
                break
        #* 如果数据集的jsonl, json, parquet格式都不存在, 则跳过
        if exist_flag == 0:
            continue
        else:
            print(f'Loading {split} dataset from {split_path}...')
        if split in ["test", 'val', 'dev']:
            split_dict[split] = Dataset(
                config, split_path,
                sample_num=config['test_sample_num'],
                random_sample=config['random_sample'],
            )
        else:
            split_dict[split] = Dataset(config, split_path)
    return split_dict

def get_decomposer(config):
    r"""Automatically select decomposer class based on config's decomposer method
    Args:
        config (dict): configuration with 'decomposer_method' key
    Returns:
        Decomposer: decomposer instance
    """
    if config['decomposer_method'] == 'ufo':
        return getattr(importlib.import_module('ufo.decomposer'), 'UFODecomposer')(config)

def get_generator(config):
    r"""Automatically select query generator class based on config's query generator method
    Args:
        config (dict): configuration with 'generator_method' key
    Returns:
        generator: Generator instance
    """
    if config['generator_method'] == 'ufo':
        return getattr(importlib.import_module('ufo.generator'), 'UFOGenerator')(config)

def get_retriever(config):
    '''
    这些retriever的输入参数为queries
    '''
    objects = {}    # retriever_name -> object
    if 'human' in config['retriever_sources']:
        retriever_human = getattr(importlib.import_module('ufo.retriever'), 'HumanRetriever')(config)
        objects['human'] = retriever_human
    if 'web' in config['retriever_sources']:
        retriever_web = getattr(importlib.import_module('ufo.retriever'), 'WebRetriever')(config)
        objects['web'] = retriever_web
    if 'knowledge' in config['retriever_sources']:
        retriever_knowledge = getattr(importlib.import_module('ufo.retriever'), 'KnowledgeRetriever')(config)
        objects['knowledge'] = retriever_knowledge
    #TODO add customized fact sources
    return objects

def get_extractor(config):
    '''extract answer from a given evidence'''
    if config['extractor_method'] == 'ufo':
        return getattr(importlib.import_module('ufo.extractor'), 'UFOExtractor')(config)

def get_verifier(config):
    if config['verifier_method'] == 'factool':
        return getattr(importlib.import_module('ufo.verifier'), 'FactoolVerifier')(config)
    elif config['verifier_method'] == 'ufo':
        return getattr(importlib.import_module('ufo.verifier'), 'UFOVerifier')(config)
    
def get_evaluator(config):
    if config['evaluator_method'] == 'avg_claim':
        return getattr(importlib.import_module('ufo.evaluator'), 'ClaimEvaluator')(config)
    if config['evaluator_method'] == 'all_claim':
        return getattr(importlib.import_module('ufo.evaluator'), 'AllClaimEvaluator')(config)
    
