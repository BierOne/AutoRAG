from evaluator import TestEvaluator
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--name',
        type=str,
        help="name for the trial directory",
        default="test_trial"
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        help="dir for loading config files",
        default="../experiments/candidate_config"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help="Output dir for storing generated single-experiment config files",
        default="/home/lyb/RAG/data/eli5_data"
    )
    parser.add_argument(
        '--project_dir',
        type=str,
        help="Output dir for storing generated logs",
        default="../experiments/eli5_runner"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    config_dir = Path(args.config_dir)
    evaluator = TestEvaluator(qa_data_path=(data_dir/'qa_sample.parquet').as_posix(),
                          corpus_data_path=(data_dir/'corpus.parquet').as_posix(),
                          project_dir=args.project_dir)
    evaluator.init_trial(trial_name=args.name, yaml_dir=config_dir.as_posix())
    evaluator.run_undone_configs()

    # for i in range(5):
    #     config_path = config_dir / f'/ollama_config_{i}.yaml'
    #     evaluator.init_runner_from_yaml(config_path)
    #     summary_df = evaluator.run_with_qa_eval()
    #     print(summary_df)

        # message = evaluator.run('I am not good. How about you?')
        # print(config_path)
        # print(message)

if __name__ == '__main__':
    main()