import json, os, hashlib, asyncio, tqdm, requests, rich, time
from typing import List, Union, Tuple
class GoogleSerperAPIWrapper:
    """Wrapper around the Serper.dev Google Search API.
    You can create a free API key at https://serper.dev.
    To use, you should have the environment variable ``SERPER_API_KEY``
    set with your API key, or pass `serper_api_key` as a named parameter
    to the constructor.
    Example:
        .. code-block:: python
            from langchain import GoogleSerperAPIWrapper
            google_serper = GoogleSerperAPIWrapper()
    """

    def __init__(self, snippet_cnt=10, api_key=None) -> None:
        self.k = snippet_cnt
        self.gl = "us"
        self.hl = "en"
        # self.serper_api_key = os.environ.get("SERPER_API_KEY", None)
        self.serper_api_key = api_key
        print(f'initialize serper api key! {self.serper_api_key}')
        assert self.serper_api_key is not None
        assert self.serper_api_key != ''

    async def _google_serper_search_results(self, session, search_term: str, gl: str, hl: str) -> dict:
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        params = {"q": search_term, "gl": gl, "hl": hl}
        async with session.post(
                "https://google.serper.dev/search", headers=headers, params=params, raise_for_status=True
        ) as response:
            return await response.json()

    def _parse_results(self, results):
        snippets = []
        # 如果有answerbox
        if results.get("answerBox"):
            answer_box = results.get("answerBox", {})
            if answer_box.get("answer"):
                return answer_box.get('answer')
                # element = {"content": answer_box.get("answer"), "source": "None"}
                # return [element]
            elif answer_box.get("snippet"):
                return answer_box.get("snippet").replace("\n", " ")
                # element = {"content": answer_box.get("snippet").replace("\n", " "), "source": "None"}
                # return [element]
            elif answer_box.get("snippetHighlighted"):
                return answer_box.get("snippetHighlighted")
                # element = {"content": answer_box.get("snippetHighlighted"), "source": "None"}
                # return [element]

        if results.get("knowledgeGraph"):
            kg = results.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                # element = {"content": f"{title}: {entity_type}", "source": "None"}
                # snippets.append(element)
                snippets.append(f"{title}: {entity_type}")
            description = kg.get("description")
            if description:
                # element = {"content": description, "source": "None"}
                snippets.append(description)
            for attribute, value in kg.get("attributes", {}).items():
                # element = {"content": f"{attribute}: {value}", "source": "None"}
                snippets.append(f"{attribute}: {value}")
        
        assert 'organic' in results, f'results: {results}'

        for _snippet_id, result in enumerate(results["organic"][: self.k]):
            if "snippet" in result:
                # element = {"content": result["snippet"], "source": result["link"]}
                # snippets.append(f'{result["snippet"]} <{result["link"]}>')
                snippets.append(f'{result["snippet"]}')
            for attribute, value in result.get("attributes", {}).items():
                # element = {"content": f"{attribute}: {value}", "source": result["link"]}
                # snippets.append(f'{attribute}: {value} <{result["link"]}>')
                snippets.append(f'{attribute}: {value}')

        if len(snippets) == 0:
            # element = {"content": "No good Google Search Result was found", "source": "None"}
            # return [element]
            return "No good Google Search Result was found"

        # keep only the first k snippets
        final_snippets = []
        for _idx, snippet in enumerate(snippets):
            final_snippets.append(f'[{_idx+1}] {snippet}')
        final_snippets = '\n'.join(final_snippets)
        return final_snippets
    
    def searches(self, ids, search_queries, gl, hl):
        if len(search_queries) == 0: return []
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        payload = json.dumps([{'q': query, 'gl': gl, 'hl': hl} for query in search_queries])
        while True:
            try:
                response = requests.request('POST', 
                    'https://google.serper.dev/search', headers=headers, data=payload, timeout=60).json()
                break
            except Exception as e:
                rich.print(f'input for search: {payload}')
                rich.print(f'raise exception! sleep for 1 second: {e}')
                time.sleep(1)
        assert len(ids) == len(response), f'got length {len(ids)} and {len(response)}\nids: {ids}\nresponse: {response}'
        results = [(_id, _res) for _id, _res in zip(ids, response)]
        return results
    

    def run(self, query_evidences):
        """Run query through GoogleSearch and parse result."""
        ids, flattened_queries = [], []
        for _id, items in enumerate(query_evidences):
            _query, _evidence = items['query'], items['evidence']
            if _evidence is None:
                ids.append(_id)
                flattened_queries.append(_query)
        ids_results = self.searches(ids, flattened_queries, gl=self.gl, hl=self.hl)
        updating_cache = []
        if len(ids_results) != 0:
            for _id, result in ids_results:
                assert query_evidences[_id]['evidence'] == None, query_evidences[_id]
                cur_query = query_evidences[_id]['query']
                snippets = self._parse_results(result)
                updating_cache.append({'query': cur_query, 'evidence': snippets})
                query_evidences[_id]['evidence'] = snippets
        return query_evidences, updating_cache
    

    
class WebRetriever:
    def __init__(self, config):
        self.url = "https://google.serper.dev/search"
        self.headers = {
            'X-API-KEY': config['retriever']['web_search']['serper_apikey'],
            'Content-Type': 'application/json'
        }
        self.snippet_cnt = config['retriever']['web_search']['snippet_cnt']
        self.cache_path = config['retriever']['web_search']['cache_path']
        self.api_key = config['retriever']['web_search']['serper_apikey']
        self.serper = GoogleSerperAPIWrapper(snippet_cnt=self.snippet_cnt, api_key=self.api_key)
        
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        print(f'web retriever got cache size: {len(self.cache)}')
    
    def get_from_cache(self, query):
        # obtain from cache
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        if query_hash in self.cache and self.cache[query_hash]['query'] == query:
            return self.cache[query_hash]['evidence']
        # otherwise, search web
        return None
        
    def save_cache(self):
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2)
        
    def update_cache(self, updating_cache):
        if isinstance(updating_cache, dict):
            updating_cache = [updating_cache]
        for cache_item in updating_cache:
            query, evidence = cache_item['query'], cache_item['evidence']
            query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
            self.cache[query_hash] = {'query': query, 'evidence': evidence}
        
    def __call__(self, input_samples_query: List[List[Union[List[str], str]]]=None, **kwargs):
        results = []
        for claims_query in tqdm.tqdm(input_samples_query, desc='Retrieving web search evid', ncols=100):
            query_evidences = []
            for _id, claim_query in enumerate(claims_query):
                if isinstance(claim_query, str):
                    query_evidences.append({'query': claim_query, 'evidence': self.get_from_cache(claim_query)})
                elif isinstance(claim_query, list):
                    query_evidences.append({'query': ' '.join(claim_query), 'evidence': self.get_from_cache(' '.join(claim_query))})
                else:
                    query_evidences.append({'query': 'No queries.', 'evidence': 'No evidence.'})
            search_outputs, updating_cache = self.serper.run(query_evidences)
            self.update_cache(updating_cache)
            self.save_cache()
            result_claims = []
            for search_output in search_outputs:
                result_claims.append([{'evidence': search_output['evidence'], 'evidence_ppl': None}])
            results.append(result_claims)
        return results
    