from configuration import *
from pathlib import Path
import yaml
import argparse
from itertools import product


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--size', '-s',
        type=int,
        help="size of sampled configurations",
        default=5
    )
    parser.add_argument(
        '--config_dir', '-d',
        type=str,
        help="Output dir for storing generated single-experiment config files",
        default="../experiments/candidate_config"
    )
    parser.add_argument(
        '--exhaustive', '-e',
        action='store_true',
    )

    return parser.parse_args()


def create_config(vectordb_node_line,
                  retriever_node_line,
                  post_retrieve_node_line,
                  extra_params={},
                  save_path="./ollama_config.yaml"
                  ):
    data = {
        'vectordb': [
            vectordb_node_line
        ],
        'node_lines': [
            retriever_node_line,
            post_retrieve_node_line
        ],
        **extra_params
    }
    # Save the modified data back to a YAML file
    with open(save_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)
    return data


def main():

    args = parse_args()
    size = args.size
    config_dir = Path(args.config_dir)

    config_path = "./configuration/config.yaml"
    retriever = RetrievalConfiguration.load_from_yaml(config_path, key="retrieval")
    retriever_node_lines = retriever.create_node_lines(size=size, exhaustive=args.exhaustive)

    vectordb = VectorDBConfiguration.load_from_yaml(config_path, key="vectordb")
    vectordb_node_lines = vectordb.create_node_lines(size=size, exhaustive=args.exhaustive)

    generator = GeneratorConfiguration.load_from_yaml(config_path, key="generator")
    generator_nodes = generator.create_nodes(size=size, exhaustive=args.exhaustive)

    prompt_maker = PromptMakerConfiguration.load_from_yaml(config_path, key="prompt_maker")
    prompt_maker_nodes = prompt_maker.create_nodes(size=size, exhaustive=args.exhaustive)


    # Generate all possible configurations
    if args.exhaustive:
        # Generate all index combinations
        all_lists = [vectordb_node_lines, retriever_node_lines, prompt_maker_nodes, generator_nodes]
        all_combinations = list(product(*[range(len(lst)) for lst in all_lists]))
        print(f"[Exhaustively] find the number of configurations: {len(all_combinations)}")
        print(f"number of vectordb configs: {len(vectordb_node_lines)}")
        print(f"number of retriever configs: {len(retriever_node_lines)}")
        print(f"number of prompt_maker configs: {len(prompt_maker_nodes)}")
        print(f"number of generator configs: {len(generator_nodes)}")

        for i, combo_idxes in enumerate(all_combinations):
            # post retrieve node line
            prompt_maker_node = prompt_maker_nodes[combo_idxes[2]]
            generator_node = generator_nodes[combo_idxes[3]]
            post_retrieve_node_line = create_lines_with_nodes("post_retrieve_node_line",
                                                           [prompt_maker_node], [generator_node])[0]
            create_config(vectordb_node_lines[combo_idxes[0]],
                          retriever_node_lines[combo_idxes[1]],
                          post_retrieve_node_line,
                          extra_params={
                            'bm25_tokenizer_list': retriever.cs.get('[bm25]bm25_tokenizer').choices \
                                if '[bm25]bm25_tokenizer' in retriever.cs else ['porter_stemmer', 'space'],
                            'strategies': {'metrics': ['meteor', 'rouge', 'bert_score']}
                          },
                          save_path=config_dir / f"ollama_config_{i}.yaml")

    else:
        post_retrieve_node_lines = create_lines_with_nodes("post_retrieve_node_line",
                                                           prompt_maker_nodes, generator_nodes)
        all_lists = zip(vectordb_node_lines, retriever_node_lines, post_retrieve_node_lines)
        print(f"[Simply] combination of the number of configurations: {len(all_lists)}")

        for i, node_line in enumerate(all_lists):
            vectordb_node_line, retriever_node_line, post_retrieve_node_line = node_line
            create_config(vectordb_node_line, retriever_node_line, post_retrieve_node_line,
                          extra_params={
                            'bm25_tokenizer_list': retriever.cs.get('[bm25]bm25_tokenizer').choices \
                                if '[bm25]bm25_tokenizer' in retriever.cs else ['porter_stemmer', 'space'],
                            'strategies': {'metrics': ['meteor', 'rouge', 'bert_score']}
                          },
                          save_path=config_dir / f"ollama_config_{i}.yaml")

if __name__ == '__main__':
    main()

# template = {
#     'vectordb': [
#         # Insert Here
#         # {
#         #     'name': 'chroma_mpnet',
#         #     'db_type': 'chroma',
#         #     'client_type': 'persistent',
#         #     'collection_name': 'huggingface_all_mpnet_base_v2',
#         #     'embedding_model': 'huggingface_all_mpnet_base_v2',
#         #     'path': '/home/lyb/RAG/experiments/chroma_mpnet'
#         # }
#     ],
#     'node_lines': [
#         # {
#         #     'node_line_name': 'retrieve_node_line',
#         #     'nodes': [
#         #         {
#         #             'node_type': 'retrieval',
#         #             'top_k': 3,
#         #             'modules': [{'module_type': 'bm25'}]
#         #         }
#         #     ]
#         # },
#         # {
#         #     'node_line_name': 'post_retrieve_node_line',
#         #     'nodes': [
#         #         {
#         #             'node_type': 'prompt_maker',
#         #             'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
#         #             'modules': [
#         #                 {
#         #                     'module_type': 'fstring',
#         #                     'prompt': 'Read the passages and answer the given question. \n Question: {query} \n Passage: {retrieved_contents} \n Answer : '
#         #                 }
#         #             ]
#         #         },
#         #
#         #         {
#         #             'node_type': 'generator',
#         #             'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
#         #             'modules': [
#         #                 {
#         #                     'module_type': 'llama_index_llm',
#         #                     'llm': 'ollama',
#         #                     'model': 'llama3',
#         #                     'temperature': 0.5,
#         #                     'batch': 1
#         #                 }
#         #             ]
#         #         }
#         #     ]
#         # }
#     ]
# }
