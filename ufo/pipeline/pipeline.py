from ufo.utils import get_decomposer, get_generator, get_retriever, get_extractor, get_verifier
import rich, tqdm, time


class BasicPipeline:
    def __init__(self, config):
        self.config = config
    def run(self, dataset):
        '''The overall inference process of a factchecking framework.'''
        raise NotImplementedError
    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""
        return dataset


    
class UFOPipeline(BasicPipeline):
    def __init__(self, config, decomposer=None, generator=None, retriever=None, verifier=None, extractor=None, online=False):
        """
        inference stage:
            claim decomposer -> query generator -> evidence extractor -> answer verifier
        """
        super().__init__(config)
        self.is_online = online
        # ===== 1. decomposer =====
        if decomposer is None:
            self.decomposer = get_decomposer(config)
        else:
            self.decomposer = decomposer

        # ===== 2. generator =====
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator
            
        # ===== 3. retriever w/ fact sources =====
        if retriever is None:
            self.retrievers = get_retriever(config)
        else:
            self.retrievers = retriever
            
        # ===== 4. answer extractor =====
        if extractor is None:
            self.extractor = get_extractor(config)
        else:
            self.extractor = extractor
        
        # ===== 5. verifier =====
        if verifier is None:
            self.verifier = get_verifier(config)
        else:
            self.verifier = verifier

        
    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""
        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)
        if do_eval:
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)
        return dataset
    
    @staticmethod
    def flatten(nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(UFOPipeline.flatten(item))
            else:
                flat_list.append(item)
        return flat_list

    @staticmethod
    def get_structure(nested_list):
        structure = []
        for item in nested_list:
            if isinstance(item, list):
                structure.append(UFOPipeline.get_structure(item))
            else:
                structure.append(None)  # 占位符，表示原始元素
        return structure
    
    @staticmethod
    def restore_shape(flat_list, original_structure):
        restored = []
        index = 0
        for item in original_structure:
            if isinstance(item, list):
                sublist_length = len(UFOPipeline.flatten(item))  # 获取子列表的展平长度
                restored_sublist = flat_list[index: index + sublist_length]
                # 如果需要保持嵌套结构，可以进一步处理
                restored.append(restored_sublist)
                index += sublist_length
            else:
                restored.append(flat_list[index])
                index += 1
        return restored
    def merge(self, claims, queries, fact_source):
        claims_with_evidences = []
        for c, q, e in zip(claims, queries, fact_source):
            claims_with_evidences.append((c, q, e))
        return claims_with_evidences
    
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        if self.is_online:
            # input must be [{}, {}, ...]
            evaluated_texts = [item['response'] for item in dataset]
            human_answers = [item['reference_answers'] for item in dataset]
        else:
            evaluated_texts = dataset.response
            human_answers = dataset.reference_answers

        #& ===== STEP 1: claim decomposition =====
        time1 = time.time()
        decomposed_claims, decomposed_claims_ppl = [], []
        batch_evaluated_texts_list = [evaluated_texts[i:i+self.config['batch_size']] for i in range(0, len(evaluated_texts), self.config['batch_size'])]
        for batch_evaluated_texts in tqdm.tqdm(batch_evaluated_texts_list, desc=f'decompose claims', ncols=100):
            batch_decomposed_claims, batch_decomposed_claims_ppl = self.decomposer(batch_evaluated_texts)
            decomposed_claims.extend(batch_decomposed_claims)
            decomposed_claims_ppl.extend(batch_decomposed_claims_ppl)
        if self.is_online:
            # online: first sample in decomposed_claims
            for _id, (online_decomposed_claims, online_decomposed_claims_ppl) in enumerate(zip(decomposed_claims, decomposed_claims_ppl)):
                dataset[_id]['output'] = [
                    {} for _ in online_decomposed_claims]
                for _id2, claims in enumerate(online_decomposed_claims):
                    dataset[_id]['output'][_id2]['claims'] = claims
                    dataset[_id]['output'][_id2]['claims_ppl'] = online_decomposed_claims_ppl
        else:
            dataset.update_output('claims', decomposed_claims)
            dataset.update_output('claims_ppl', decomposed_claims_ppl)
        rich.print(f'STEP 1 - Claims: {decomposed_claims}')
        rich.print(f'STEP 1 - Claim Decomposition: {time.time()-time1:.2f} (s)')
        
        #& ===== STEP 2: query generation =====
        #& flatten all claims into one batch, and after that, restore into original shapes
        time1 = time.time()
        flatten_claims = self.flatten(decomposed_claims)
        structure = self.get_structure(decomposed_claims)
        flatten_queries, flatten_queries_ppl = [], []
        batch_flatten_claims_list = [flatten_claims[i:i+self.config['batch_size']] for i in range(0, len(flatten_claims), self.config['batch_size'])]
        for batch_flatten_claims in tqdm.tqdm(batch_flatten_claims_list, desc=f'generate queries', ncols=100):
            batch_flatten_queries, batch_flatten_queries_ppl = self.generator(batch_flatten_claims)
            flatten_queries.extend(batch_flatten_queries)
            flatten_queries_ppl.extend(batch_flatten_queries_ppl)
        queries = self.restore_shape(flatten_queries, structure)
        queries_ppl = self.restore_shape(flatten_queries_ppl, structure)
        if self.is_online:
            for _id, (online_queries, online_queries_ppl) in enumerate(zip(queries, queries_ppl)):
                for _id2, query in enumerate(online_queries):
                    dataset[_id]['output'][_id2]['queries'] = query
                    dataset[_id]['output'][_id2]['queries_ppl'] = online_queries_ppl
        else: 
            dataset.update_output('queries', queries)
            dataset.update_output('queries_ppl', queries_ppl)
        rich.print(f'STEP 2 - Queries: {queries}')
        rich.print(f'STEP 2 - Query Generation: {time.time()-time1:.2f} (s)')
        
        #& ===== STEP 3: retriever (with several sources) =====
        evidence_list = {}
        for source_id, (source, custom_retriever) in enumerate(self.retrievers.items()):
            source_name = f'{source}_sources'
            rich.print(f'STEP 3 [{source_id+1}/{len(self.retrievers.keys())}] - {source_name} retrieval')
            time1 = time.time()
            evidence = custom_retriever(queries, human_answers=human_answers)
            evidence_list[source_name] = evidence
            if self.is_online:
                for _id, online_evidence in enumerate(evidence):
                    for _id2, _evid in enumerate(online_evidence):
                        dataset[_id]['output'][_id2][f'{source}_sources'] = _evid
            else:
                dataset.update_output(source_name, evidence)
            rich.print(f'STEP 3 [{source_id+1}/{len(self.retrievers.keys())}] - {source_name} retrieval: {time.time()-time1:.2f} (s)')
            
        #& ===== STEP 4: extractor =====
        #& get the answer from each evidence
        extraction_list = {}
        for source_id, source in enumerate(self.retrievers.keys()):
            source_name = f'{source}_sources'
            rich.print(f'STEP 4 [{source_id+1}/{len(self.retrievers.keys())}] - {source_name} extraction')
            extractions = []
            time1 = time.time()
            for sample_decomposed_claims, sample_queries, sample_evidence in tqdm.tqdm(list(zip(decomposed_claims, queries, evidence_list[source_name])), desc=f'extract {source_name}', ncols=100):
                assert len(sample_decomposed_claims) == len(sample_queries) == len(sample_evidence)
                input_extraction = list(zip(*[sample_decomposed_claims, sample_queries, sample_evidence]))
                #* claim-level的抽取答案 [(c1, q1, e1), (c2, q2, e2), ...]
                sample_extractions = self.extractor(input_extraction)
                extractions.append(sample_extractions)
            rich.print(f'STEP 4 [{source_id+1}/{len(self.retrievers.keys())}] - {source_name} extraction: {time.time()-time1:.2f} (s)')
            extraction_list[source_name] = extractions
            if self.is_online:
                for _id, online_extraction in enumerate(extractions):
                    for _id2, extraction in enumerate(online_extraction):
                        dataset[_id]['output'][_id2][f'{source}_extractions'] = extraction
            else:
                dataset.update_output(f'{source}_extractions', extractions)

                
        
        #& ===== STEP 5: verifier =====
        #& verify所有可用的事实源
        for source_id, source in enumerate(self.retrievers.keys()):
            source_name = f'{source}_sources'
            rich.print(f'STEP 5 [{source_id+1}/{len(self.retrievers.keys())}] - {source_name} verification')
            details, details_ppl = [], []
            time1 = time.time()
            for sample_decomposed_claims, sample_queries, sample_evidence, sample_answers in tqdm.tqdm(list(zip(decomposed_claims, queries, evidence_list[source_name], extraction_list[source_name])), desc=f'verify {source_name}', ncols=100):
                assert len(sample_decomposed_claims) == len(sample_queries) == len(sample_evidence) == len(sample_answers)
                input_verification = list(zip(*[sample_decomposed_claims, sample_queries, sample_evidence, sample_answers]))
                #* claim-level的评估 [(c1, q1, e1, a1), (c2, q2, e2, a2), ...]
                sample_details = self.verifier(input_verification)
                details.append(sample_details)
            rich.print(f'STEP 5 [{source_id+1}/{len(self.retrievers.keys())}] - {source_name} verification: {time.time()-time1:.2f} (s)')
            if self.is_online:
                for _id, online_details in enumerate(details):
                    for _id2, detail in enumerate(online_details):
                        dataset[_id]['output'][_id2][f'{source}_details'] = detail
            else:
                dataset.update_output(f'{source}_details', details)
        
        return dataset