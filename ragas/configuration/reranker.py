from .base import *


STATIC_PARAMS = {
"upr": {
        "use_bf16": False,
        "prefix_prompt": "Passage: ", # start prompt of the paragraph
        "suffix_prompt": "Please write a question based on this passage." # end prompt
    },
"tart": {"instruction":"Find passage to answer given question",
         "batch": 64
         },
"monot5": { 
    "model_name": "castorini/monot5-base-msmarco",
        # "castorini/monot5-base-msmarco-10k",
        # "castorini/monot5-large-msmarco",
        # "castorini/monot5-large-msmarco-10k",
        # "castorini/monot5-base-med-msmarco",
        # "castorini/monot5-3b-med-msmarco",
        # "castorini/monot5-3b-msmarco-10k",
        # "unicamp-dl/mt5-base-en-msmarco",
        # "unicamp-dl/ptt5-base-pt-msmarco-10k-v2",
        # "unicamp-dl/ptt5-base-pt-msmarco-100k-v2",
        # "unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2",
        # "unicamp-dl/mt5-base-en-pt-msmarco-v2",
        # "unicamp-dl/mt5-base-mmarco-v2",
        # "unicamp-dl/mt5-base-en-pt-msmarco-v1",
        # "unicamp-dl/mt5-base-mmarco-v1",
        # "unicamp-dl/ptt5-base-pt-msmarco-10k-v1",
        # "unicamp-dl/ptt5-base-pt-msmarco-100k-v1",
        # "unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1",
        # "unicamp-dl/mt5-3B-mmarco-en-pt",
        # "unicamp-dl/mt5-13b-mmarco-100k",
    
    "batch" : 64
},
"koreranker":{
    "batch":64
    },
"cohere_reranker": {
    "api_key": "your_cohere_api_key", # OpenAI API key. You can also set this to env variable OPENAI_API_KEY.
    "batch": 64,
    "model": "rerank-multilingual-v2.0", # The model you can choose
    # "rerank-v3.5",
    # "rerank-english-v3.0",
    # "rerank-multilingual-v3.0",
},
"rankgpt": { # must start with openai model or with other llm init from generator
    "verbose": False,
    "batch": 8
},
"jina_reranker": {
  "api_key": "your_jina_api_key",
  "batch": 16,
  "model": "jina-reranker-v1-base-en", # The model you can choose
#   "jina-colbert-v1-en",
},
 "colbert_reranker": {
  "batch": 64,
  "model_name": "colbert-ir/colbertv2.0"
},
"sentence_transformer_reranker":{
  "batch": 32,
  "model_name": "cross-encoder/ms-marco-MiniLM-L-2-v2"
},
"flag_embedding_reranker": {
  "batch": 32,
  "model_name": "BAAI/bge-reranker-large",
  "use_fp16" : False,
},
"flag_embedding_llm_reranker": {
  "batch": 32,
  "model_name": "BAAI/bge-reranker-v2-gemma",
  "use_fp16" : False,
},
"time_reranker":None,
"openvino_reranker": {
  "batch": 32,
  "model": "BAAI/bge-reranker-large"
},
"voyageai_reranker": {
  "api_key": "your_voyageai_api_key",
  "model": "rerank-2", # The model you can choose
  # "rerank-2-lite",
  "truncation": True
},
"mixedbreadai_reranker": {
  "api_key": "your_mixedbread_api_key"
},
"flashrank_reranker": {
  "batch": 32,
  "model": "ms-marco-MiniLM-L-12-v2"
}
}

class RerankerConfiguration(BaseConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)
        self.para = self.translate(config)

    def build(self, config=None) -> ConfigurationSpace | None:
        cs = ConfigurationSpace(
            space={
                "reranker_name": CategoricalHyperparameter("reranker_name",
                                                            ['upr', 'tart', 'monot5', 'koreranker', 'cohere_reranker', 'rankgpt', 'jina_reranker', 'colbert_reranker', 'sentence_transformer_reranker', 'flag_embedding_reranker', 'flag_embedding_llm_reranker', 'time_reranker', 'openvino_reranker', 'voyageai_reranker', 'mixedbreadai_reranker', 'flashrank_reranker'],
                                                            weights=[1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1]),
            }
        )
        # Add general hyperparameters
        top_k = UniformIntegerHyperparameter("top_k", lower=1, upper=10, default_value=5)
        temperature = CategoricalHyperparameter("temperature", choices=[0.5, 1.0, 1.5])
        cs.add([temperature, top_k])
        cs.add(InCondition(cs["temperature"], cs["reranker_name"], ["rankgpt"]),) # only rankgot has the editable parameter
        return cs

    def sampling(self, size: Optional[int] = 1) -> Union[Configuration, List[Configuration]]:
        return self.cs.sample_configuration(size)

    def translate(self, config) -> Dict:
        return self.cs.sample_configuration(size)
