import os
from pathlib import Path
from datasets import load_dataset


main_dir = Path("/home/lyb")
data_dir = main_dir / "RAG/data/eli5_data"

from autorag.evaluator import Evaluator
evaluator = Evaluator(qa_data_path=(data_dir/'qa_sample.parquet').as_posix(),
                      corpus_data_path=(data_dir/'corpus.parquet').as_posix(),
                      project_dir='./eli5_runs')

for i in range(5):
    evaluator.start_trial(f'./candidate_config/ollama_config_{i}.yaml', skip_validation=True)

