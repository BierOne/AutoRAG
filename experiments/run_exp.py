# import os
# from pathlib import Path
# from datasets import load_dataset


# main_dir = Path("/home/lyb")
# data_dir = main_dir / "RAG/data/eli5_data"

# from autorag.evaluator import Evaluator
# evaluator = Evaluator(qa_data_path=(data_dir/'qa_sample.parquet').as_posix(), corpus_data_path=(data_dir/'corpus.parquet').as_posix(),
#                       project_dir='./eli5_runs')

# for i in range(9): 
#     evaluator.start_trial(f'./candidate_config/ollama_config_{i}.yaml', skip_validation=True)


import numpy as np

board_size, moves = 4, [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
is_valid = lambda x, y: 0 <= x < board_size and 0 <= y < board_size

def build_transition_matrix():
    P = np.zeros((16, 16))
    for x in range(4):
        for y in range(4):
            i = x * 4 + y
            if (x, y) == (3, 3): P[i][i] = 1; continue
            valid_moves = [(x + dx) * 4 + (y + dy) for dx, dy in moves if is_valid(x + dx, y + dy)]
            for j in valid_moves: P[i][j] = 1 / len(valid_moves)
    return P

def solve_expectation():
    P, I = build_transition_matrix(), np.eye(16)
    Q, b = I - P, np.ones(16)
    Q[-1, -1], b[-1] = 1, 0
    return np.linalg.solve(Q, b)[0]

print(f"首次到达右下角所需步数的期望值为：{solve_expectation():.4f}")

