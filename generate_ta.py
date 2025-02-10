import os
import jsonlines
import logging
import time
import multiprocessing
import math
from tqdm import tqdm
from datasets import load_dataset
from typing import List
from openai import OpenAI
from datasets import load_dataset
from util import PrompterCot, Prompter
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


def _request_topic(client: OpenAI, queue_prompt, queue_response):
    while True:
        idx, prompt = queue_prompt.get()
        completion = client.chat.completions.create(
            model="llama",
            n=25,
            max_tokens=16,
            temperature=1.2,
            messages=[{"role": "user", "content": prompt}]
        )
        res = [c.message.content for c in completion.choices]
        temp = []
        for r in res:
            if ":" in r:
                r = r.split(":")[1].strip()
            temp.append(r)
        res = temp
        queue_response.put((idx, res))


def _request_question(client: OpenAI, queue_prompt, queue_response):
    while True:
        idx, prompt, request_num = queue_prompt.get()
        completion = client.chat.completions.create(
            model="llama",
            n=request_num,
            max_tokens=28,
            temperature=1.2,
            messages=[{"role": "user", "content": prompt}]
        )
        res = [c.message.content for c in completion.choices]
        temp = []
        for r in res:
            if ":" in r:
                r = r.split(":")[1].strip()
            temp.append(r)
        res = temp
        queue_response.put((idx, res))


class Generate:

    def __init__(self, data_dir):
        cache_dir = "./generated_query_llama_ta_8b"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.dataset_name = data_dir.split("/")[-1]
        corpus = load_dataset('json', data_files=f"{data_dir}/corpus.jsonl")["train"]
        self.data = [f"{x['title']}: {x['text']}" for x in corpus]
        self.id = [x["_id"] for x in corpus]
        self.save_path = f"./{cache_dir}/{self.dataset_name}_generated_queries.jsonl"
        self.topic_save_path = f"./{cache_dir}/{self.dataset_name}_generated_topics.jsonl"
        self.prompter_cot = PrompterCot(self.dataset_name)
        self.prompter = Prompter(self.dataset_name)
        logging.info(f"Dataset: {self.dataset_name}, {len(self.data)} passages")

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
            topics, questions = self._generate(process_data[i:i+batch_size])
            with jsonlines.open(self.save_path, mode='a') as f:
                save_data = [{"id": i, "generated_queries": r} for i, r in zip(process_id[i:i+batch_size], questions)]
                f.write_all(save_data)
            with jsonlines.open(self.topic_save_path, mode='a') as f:
                save_data = [{"id": i, "generated_topics": t} for i, t in zip(process_id[i:i+batch_size], topics)]
                f.write_all(save_data)


class GenerateByAPI(Generate):

    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.dataset_name = data_dir.split("/")[-1]

    def _generate(self, passage_list: List[str]) -> List[list[str]]:
        temp = []
        for passage in passage_list:
            tokens = tokenizer.encode(passage, add_special_tokens=False)
            if len(tokens) > 6*1024:
                temp.append(tokenizer.decode(tokens[:6*1024]))
            else:
                temp.append(passage)
        passage_list = temp

        while not queue_prompt.empty():
            queue_prompt.get()
        while not queue_topic.empty():
            queue_topic.get()
        while not queue_response.empty():
            queue_response.get()

        # generate topics
        total = 0
        topics = [[] for _ in range(len(passage_list))]
        for i, passage in enumerate(passage_list):
            prompt = self.prompter_cot.topic_prompt(passage)
            queue_topic.put((i, prompt))
            total += 1

        for _ in tqdm(range(total), desc=f"Request {self.dataset_name} topics", total=total, mininterval=TQDM_MIN_INTERVAL):
            idx, gen_data = queue_response.get()
            gen_data = list(set(gen_data))
            gen_data = [d for d in gen_data if len(d) > 6]
            topics[idx] = gen_data
        
        # generate questions
        total = 0
        questions = [[] for _ in range(len(passage_list))]
        request_num = 4
        for i, passage in enumerate(passage_list):
            for t in topics[i]:
                prompt = self.prompter_cot.cot_prompt(passage, t)
                queue_prompt.put((i, prompt, request_num))
                total += 1
            if len(topics[i])*request_num < 100:
                prompt = self.prompter.prompt_fn(passage)
                temp_request_num = 100 - request_num*len(topics[i])
                queue_prompt.put((i, prompt, temp_request_num))
                total += 1

        for _ in tqdm(range(total), desc=f"Request {self.dataset_name} questions", total=total, mininterval=TQDM_MIN_INTERVAL):
            idx, gen_data = queue_response.get()
            questions[idx] += gen_data

        return topics, questions



if __name__ == '__main__':

    url_list = ["http://localhost:8080/v1"]
    api_key_list = ["api-key"]
    client_list = [OpenAI(
        base_url=url,
        api_key=api_key,
    ) for url, api_key in zip(url_list, api_key_list)]
    process_num_pergpu = 80
    topic_process_num_pergpu = 30
    process_num = len(client_list) * process_num_pergpu
    topic_process_num = len(client_list) * topic_process_num_pergpu

    queue_prompt = multiprocessing.Queue()
    queue_topic = multiprocessing.Queue()
    queue_response = multiprocessing.Queue()
    for i in range(process_num):
        multiprocessing.Process(target=_request_question, args=(client_list[i%len(client_list)], queue_prompt, queue_response), daemon=True).start()
    for i in range(topic_process_num):
        multiprocessing.Process(target=_request_topic, args=(client_list[i%len(client_list)], queue_topic, queue_response), daemon=True).start()



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


