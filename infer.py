import os
import torch
import jsonlines
import numpy as np
import torch.multiprocessing as mp
import warnings
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn import mixture
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader
warnings.simplefilter(action='ignore', category=FutureWarning)

TQDM_MIN_INTERVAL = 1
MODEL_PATH = "/PATH/pretrainedmodel/all-MiniLM-L12-v2"
GENERATE_QUERY_PATH = ["generated_query_llama_ta_8b", "generated_query_llama_mg_8b", "generated_query_llama_zs_8b"]


def timeit(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time cost: {time.time()-start}")
        return result
    return wrapper

def gpu_encode_passage(rank, model_path, passages_queue: mp.Queue, result_queue: mp.Queue, lock):

    batch_size = 600
    model = SentenceTransformer(model_path, device=f"cuda:{rank}").eval()
    while True:
        idx_list = []
        passages_len = []
        passages = []
        while len(passages) < batch_size:
            lock.acquire()
            if not passages_queue.empty():
                idx, passage = passages_queue.get()
                lock.release()
            else:
                lock.release()
                break
            idx_list.append(idx)
            passages_len.append(len(passage))
            passages.extend([p.strip() for p in passage]) # remove leading and trailing whitespaces
        if len(passages) == 0:
            continue

        embeddings = model.encode(passages, show_progress_bar=False)
        
        offset = 0
        for idx, length in zip(idx_list, passages_len):
            result_queue.put((idx, embeddings[offset:offset+length]))
            offset += length
            


def cpu_gmm_and_search(tqdm_min_interval, query_embedding, topk, max_loop, result_queue: mp.Queue):
    score = []
    score_index = []
    total_exception = 0
    for _ in tqdm(range(max_loop), desc="Calculating score", total=max_loop, mininterval=tqdm_min_interval):
        index, embedding = result_queue.get()

        gmm = mixture.GaussianMixture(n_components=4, covariance_type='diag', random_state=42).fit(embedding)
        min_bic = gmm.bic(embedding)
        passage_embedding = gmm.means_
        for c in range(5,11):
            try:
                gmm = mixture.GaussianMixture(n_components=c, covariance_type='diag', random_state=42).fit(embedding)
                cur_bic = gmm.bic(embedding)
                if cur_bic < min_bic:
                    min_bic = cur_bic
                    passage_embedding = gmm.means_
            except Exception as e:
                print(e)
                total_exception += 1
                print(f"Total exception: {total_exception}")
                break

        passage_embedding = passage_embedding / np.linalg.norm(passage_embedding, axis=1, keepdims=True)


        score_p = np.matmul(query_embedding, passage_embedding.T)
        score_p = np.max(score_p, axis=1, keepdims=True)

        score.append(score_p)
        score_index.append(np.ones_like(score_p, dtype=np.int32)*index)

        if len(score) >= 500:
            score = np.concatenate(score, axis=1)
            score_index = np.concatenate(score_index, axis=1)
            top_index = np.argsort(-score, axis=1)[:,:topk]
            score = [np.take_along_axis(score, top_index, axis=1)]
            score_index = [np.take_along_axis(score_index, top_index, axis=1)]
    
    score = np.concatenate(score, axis=1)
    score_index = np.concatenate(score_index, axis=1)
    top_index = np.argsort(-score, axis=1)[:,:topk]
    score = np.take_along_axis(score, top_index, axis=1)
    index = np.take_along_axis(score_index, top_index, axis=1)
    print(f"Total exception: {total_exception}")

    return score, index


class BeirInfer:

    def __init__(self, data_dir):
        self.dataset_name = data_dir.split("/")[-1]

        self.passage_generated_queries = []
        self.passage_id_list = []
        _id2generated_queries = {}
        for path in GENERATE_QUERY_PATH:
            with jsonlines.open(f"{path}/{self.dataset_name}_generated_queries.jsonl") as reader:
                for obj in reader:
                    if obj["id"] not in _id2generated_queries:
                        _id2generated_queries[obj["id"]] = []
                    _id2generated_queries[obj["id"]] += obj["generated_queries"]
        for _id, generated_queries in _id2generated_queries.items():
            self.passage_id_list.append(_id)
            self.passage_generated_queries.append(generated_queries)


        queries = load_dataset('json', data_files=f"{data_dir}/queries.jsonl")["train"]
        self.query_id_list = []
        self.query_list = []
        for query in queries:
            self.query_id_list.append(query["_id"])
            self.query_list.append(query["text"])

        _, _, self.qrel_dict = GenericDataLoader(data_dir).load(split="test")


    def encode_query(self, query_list):
        model = SentenceTransformer(MODEL_PATH, device="cuda").eval()

        batch_size = 512
        query_embedding_list = []
        for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding queries", mininterval=TQDM_MIN_INTERVAL):
            input = query_list[i:i + batch_size]
            with torch.no_grad():
                outputs = model.encode(input, show_progress_bar=False)
                query_embedding_list.append(outputs)
        query_embedding_list = np.concatenate(query_embedding_list, axis=0)
        del model
        return query_embedding_list
    
    
    def calc_metric(self, qrel_dict, top_k_dict):
        '''
        qrel_dict: {query_id: {passage_id: relevance}}
        top_k_dict: {query_id: [passage_id]}
        '''
        user_results = {query_id: {passage_id: idx*10 for idx,passage_id in enumerate(passage_id_list[::-1])} 
                        for query_id, passage_id_list in top_k_dict.items()}
        res = EvaluateRetrieval.evaluate(qrel_dict, user_results, k_values=[10])

        return {
            "dataset": self.dataset_name,
            "res": res
        }
    
    # @timeit
    def search_multiprocess(self, topk=10):
        total_passage = len(self.passage_generated_queries)

        while not passages_queue.empty():
            passages_queue.get()
        while not result_queue.empty():
            result_queue.get()
        for i, passage in tqdm(enumerate(self.passage_generated_queries), desc="Put passages into queue", mininterval=TQDM_MIN_INTERVAL):
            passages_queue.put((i, passage))
        
        query_embedding = self.encode_query(self.query_list)
        score_topk, score_topk_index = cpu_gmm_and_search(TQDM_MIN_INTERVAL, query_embedding, topk, total_passage, result_queue)
        
        top_k_dict = {query_id: [self.passage_id_list[idx] for idx in score_topk_index[i]]
                       for i, query_id in enumerate(self.query_id_list)}
        return self.calc_metric(self.qrel_dict, top_k_dict)
    
    @timeit
    def search(self, topk=10):
        return self.search_multiprocess(topk)



if __name__ == "__main__":
    mp.set_start_method("spawn")
    passages_queue = mp.Queue()
    result_queue = mp.Queue()
    lock = mp.Lock()
    for i in range(torch.cuda.device_count()):
        mp.Process(target=gpu_encode_passage, args=(i, MODEL_PATH, passages_queue, result_queue, lock), daemon=True).start()
        mp.Process(target=gpu_encode_passage, args=(i, MODEL_PATH, passages_queue, result_queue, lock), daemon=True).start()



    data_dir = "/PATH/BEIR/nfcorpus"
    nf = BeirInfer(data_dir)
    print(nf.search())

    data_dir = "/PATH/BEIR/scifact"
    scifact = BeirInfer(data_dir)
    print(scifact.search())

    data_dir = "/PATH/BEIR/scidocs"
    scidocs = BeirInfer(data_dir)
    print(scidocs.search())

    data_dir = "/PATH/BEIR/arguana"
    arguana = BeirInfer(data_dir)
    print(arguana.search())

    data_dir = "/PATH/BEIR/fiqa"
    fiqa = BeirInfer(data_dir)
    print(fiqa.search())

    data_dir = "/PATH/BEIR/trec-covid"
    trec_covid = BeirInfer(data_dir)
    print(trec_covid.search())


