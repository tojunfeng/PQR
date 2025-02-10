import os
import jsonlines
import logging
import time
import multiprocessing
from tqdm import tqdm
from datasets import load_dataset
from typing import List
from openai import OpenAI
from datasets import load_dataset
from util import Prompter
from transformers import AutoTokenizer

TQDM_MIN_INTERVAL = 300
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tokenizer = AutoTokenizer.from_pretrained("/PATH/huggingface/Meta-Llama-3.1-8B-Instruct")

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.info('%r  %2.3f s' % (method.__name__, (te - ts)))
        return result

    return timed


def _request(client, queue_prompt, queue_response):
    while True:
        idx, prompt = queue_prompt.get()
        completion = client.chat.completions.create(
            model="llama",
            n=100,
            max_tokens=28,
            temperature=1.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        res = [c.message.content for c in completion.choices]
        queue_response.put((idx, res))



class Generate:

    def __init__(self, data_dir):
        cache_dir = "./generated_query_llama_zs_8b"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.dataset_name = data_dir.split("/")[-1]
        corpus = load_dataset('json', data_files=f"{data_dir}/corpus.jsonl")["train"]
        self.data = [f"{x['title']}: {x['text']}" for x in corpus]
        self.id = [x["_id"] for x in corpus]
        self.save_path = f"./{cache_dir}/{self.dataset_name}_generated_queries.jsonl"
        logging.info(f"Dataset: {self.dataset_name}, {len(self.data)} passages")

        self.prompt_fn = Prompter(self.dataset_name).prompt_fn
    

    def _generate(self, passage_list: List[str]) -> List[list[str]]:
        '''
        passage_list: List[str], a list of passages
        return: List[List[str]], a list of generated queries
        '''
        raise NotImplementedError


    @timeit
    def generate(self):
        cache_passage_id = []
        if os.path.exists(self.save_path):
            with jsonlines.open(self.save_path, "r") as f:
                for line in f:
                    cache_passage_id.append(line["id"])
        cache_passage_id = set(cache_passage_id)

        process_data = []
        process_id = []
        for _id, passage in zip(self.id, self.data):
            if _id not in cache_passage_id:
                process_data.append(passage)
                process_id.append(_id)
        print(f"Skip {len(self.data) - len(process_data)} passages")

        batch_size = 1000
        for i in tqdm(range(0, len(process_data), batch_size), desc=f"Generate {self.dataset_name}", total=len(process_data)//batch_size):
            res = self._generate(process_data[i:i+batch_size])
            save_data = [{"id": i, "generated_queries": r} for i, r in zip(process_id[i:i+batch_size], res)]
            with jsonlines.open(self.save_path, mode='a') as f:
                f.write_all(save_data)


class GenerateByAPI(Generate):

    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.dataset_name = data_dir.split("/")[-1]

    def _generate(self, passage_list: List[str]) -> List[list[str]]:
        while not queue_prompt.empty():
            queue_prompt.get()
        while not queue_response.empty():
            queue_response.get()

        res = [[] for _ in range(len(passage_list))]
        for i, passage in enumerate(passage_list):
            tokens = tokenizer.encode(passage, add_special_tokens=False)
            if len(tokens) > 6*1024:
                passage = tokenizer.decode(tokens[:6*1024])
            prompt = self.prompt_fn(passage)
            queue_prompt.put((i, prompt))
        total = len(passage_list)
        
        for _ in tqdm(range(total), desc=f"Request {self.dataset_name}", total=total, mininterval=TQDM_MIN_INTERVAL):
            idx, gen_data = queue_response.get()
            res[idx] += gen_data

        return res



if __name__ == '__main__':

    url_list = ["http://localhost:8080/v1"]
    api_key_list = ["api-key"]
    client_list = [OpenAI(
        base_url=url,
        api_key=api_key,
    ) for url, api_key in zip(url_list, api_key_list)]
    process_num_pergpu = 12
    process_num = len(client_list) * process_num_pergpu
    queue_prompt = multiprocessing.Queue()
    queue_response = multiprocessing.Queue()
    for i in range(process_num):
        multiprocessing.Process(target=_request, args=(client_list[i%len(client_list)], queue_prompt, queue_response), daemon=True).start()



    data_dir = "/PATH/BEIR/nfcorpus"
    generator = GenerateByAPI(data_dir)
    logging.info(f"Start generating queries for {generator.__class__.__name__}")
    res = generator.generate()
    
    data_dir = "/PATH/BEIR/scifact"
    generator = GenerateByAPI(data_dir)
    logging.info(f"Start generating queries for {generator.__class__.__name__}")
    res = generator.generate()

    data_dir = "/PATH/BEIR/scidocs"
    generator = GenerateByAPI(data_dir)
    logging.info(f"Start generating queries for {generator.__class__.__name__}")
    res = generator.generate()

    data_dir = "/PATH/BEIR/arguana"
    generator = GenerateByAPI(data_dir)
    logging.info(f"Start generating queries for {generator.__class__.__name__}")
    res = generator.generate()

    data_dir = "/PATH/BEIR/fiqa"
    generator = GenerateByAPI(data_dir)
    logging.info(f"Start generating queries for {generator.__class__.__name__}")
    res = generator.generate()

    data_dir = "/PATH/BEIR/trec-covid"
    generator = GenerateByAPI(data_dir)
    logging.info(f"Start generating queries for {generator.__class__.__name__}")
    res = generator.generate()


