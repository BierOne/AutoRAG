from abc import abstractmethod
from typing import List, Tuple, Union
from .util import load_yaml_config
import os
from pathlib import Path
from typing import Optional, Dict
from collections.abc import ItemsView, Iterable, Iterator
from typing import IO, TYPE_CHECKING, Any, Literal, Mapping, Sequence, overload
from ConfigSpace.util import generate_grid

import pandas as pd
import numpy as np
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, \
    NotEqualsCondition, Configuration, InCondition
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant



def create_lines_with_nodes(node_line_name: str = "post_retrieve_node_line", *nodes) -> List[Dict]:

    node_lines = [{
        "node_line_name": node_line_name,
        "nodes": list(node_group)
    } for node_group in zip(*nodes)]
    return node_lines


def summary_df_to_yaml(summary_df: pd.DataFrame, config_dict: Dict) -> Dict:
    """
    Convert trial summary dataframe to config yaml file.

    :param summary_df: The trial summary dataframe of the evaluated trial.
    :param config_dict: The yaml configuration dict for the pipeline.
        You can load this to access trail_folder/config.yaml.
    :return: Dictionary of config yaml file.
        You can save this dictionary to yaml file.
    """

    # summary_df columns : 'node_line_name', 'node_type', 'best_module_filename',
    #                      'best_module_name', 'best_module_params', 'best_execution_time'
    node_line_names = extract_node_line_names(config_dict)
    node_strategies = extract_node_strategy(config_dict)
    strategy_df = pd.DataFrame(
        {
            "node_type": list(node_strategies.keys()),
            "strategy": list(node_strategies.values()),
        }
    )
    summary_df = summary_df.merge(strategy_df, on="node_type", how="left")
    summary_df["categorical_node_line_name"] = pd.Categorical(
        summary_df["node_line_name"], categories=node_line_names, ordered=True
    )
    summary_df = summary_df.sort_values(by="categorical_node_line_name")
    grouped = summary_df.groupby("categorical_node_line_name", observed=False)

    node_lines = [
        {
            "node_line_name": node_line_name,
            "nodes": [
                {
                    "node_type": row["node_type"],
                    "strategy": row["strategy"],
                    "modules": [
                        {
                            "module_type": row["best_module_name"],
                            **row["best_module_params"],
                        }
                    ],
                }
                for _, row in node_line.iterrows()
            ],
        }
        for node_line_name, node_line in grouped
    ]
    return {"node_lines": node_lines}


def extract_best_config(trial_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Extract the optimal pipeline from the evaluated trial.

    :param trial_path: The path to the trial directory that you want to extract the pipeline from.
        Must already be evaluated.
    :param output_path: Output path that pipeline yaml file will be saved.
        Must be .yaml or .yml file.
        If None, it does not save YAML file and just returns dict values.
        Default is None.
    :return: The dictionary of the extracted pipeline.
    """
    summary_path = os.path.join(trial_path, "summary.csv")
    if not os.path.exists(summary_path):
        raise ValueError(f"summary.csv does not exist in {trial_path}.")
    trial_summary_df = load_summary_file(
        summary_path, dict_columns=["best_module_params"]
    )
    config_yaml_path = os.path.join(trial_path, "config.yaml")
    with open(config_yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    yaml_dict = summary_df_to_yaml(trial_summary_df, config_dict)
    yaml_dict["vectordb"] = extract_vectordb_config(trial_path)
    if output_path is not None:
        with open(output_path, "w") as f:
            yaml.safe_dump(yaml_dict, f)
    return yaml_dict


def extract_vectordb_config(trial_path: str) -> List[Dict]:
    # get vectordb.yaml file
    project_dir = pathlib.PurePath(os.path.realpath(trial_path)).parent
    vectordb_config_path = os.path.join(project_dir, "resources", "vectordb.yaml")
    if not os.path.exists(vectordb_config_path):
        raise ValueError(f"vectordb.yaml does not exist in {vectordb_config_path}.")
    with open(vectordb_config_path, "r") as f:
        vectordb_dict = yaml.safe_load(f)
    result = vectordb_dict.get("vectordb", [])
    if len(result) != 0:
        return result
    # return default setting of chroma
    return [
        {
            "name": "default",
            "db_type": "chroma",
            "client_type": "persistent",
            "embedding_model": "openai",
            "collection_name": "openai",
            "path": os.path.join(project_dir, "resources", "chroma"),
        }
    ]



def parse_hyperparameters_samples(hp_config: Mapping[str, Any], module_type: Optional[str] = None) -> Mapping[str, Any]:
    if module_type is None:
        try:
            module_type = hp_config['module_type']
        except KeyError:
            module_type = hp_config['db_type']

    prefix = f"[{module_type}]"
    parsed_dict = {}
    for key, value in hp_config.items():
        if key.startswith(prefix):
            # Remove the prefix '[module_type]'
            new_key = key[len(prefix):]
            parsed_dict[new_key] = value.item() if isinstance(value, np.generic) else value
        else:
            parsed_dict[key] = value.item() if isinstance(value, np.generic) else value

    return parsed_dict

def parse_hyperparameters_from_dict(
    items: Mapping[str, Any],
) -> Iterator[Hyperparameter]:
    for name, hp in items.items():
        # Anything that is a Hyperparameter already is good
        # Note that we discard the key name in this case in favour
        # of the name given in the dictionary
        if isinstance(hp, Hyperparameter):
            yield hp

        # Tuples are bounds, check if float or int
        elif isinstance(hp, tuple):
            if len(hp) != 2:
                raise ValueError(f"'{name}' must be (lower, upper) bound, got {hp}")

            lower, upper = hp
            if isinstance(lower, float):
                yield UniformFloatHyperparameter(name, lower, upper)
            else:
                yield UniformIntegerHyperparameter(name, lower, upper)

        # Lists are categoricals
        elif isinstance(hp, list):
            if len(hp) == 0:
                raise ValueError(f"Can't have empty list for categorical {name}")

            yield CategoricalHyperparameter(name, hp)
        else:
            # It's a constant
            yield Constant(name, hp)


class BaseConfiguration:

    def __init__(
        self,
        config: Dict,
        project_dir: Optional[str] = None
    ):
        self.config = config
        self.project_dir = os.getcwd() if project_dir is None else project_dir
        self.cs = None

    @classmethod
    def load_from_yaml(cls, path: str | Path | IO[str], key: Optional[str] = None, project_dir: Optional[str] = None, **kwargs: Any,) -> ConfigurationSpace:
        """Decode a serialized configuration space from a yaml file.
        """
        config = load_yaml_config(path, **kwargs)

        if key is not None:
            config = config[key]

        print(config)
        return cls(config)


    @staticmethod
    def load_static_params(module_type: str) -> Dict:
        """
        Get the default parameters for the module.
        These would not be used for hyperparameter optimization.
        """
        pass



    def sampling(self, size: Optional[int] = 1, exhaustive = False) -> Union[Configuration, List[Configuration]]:
        """
        Sample the configuration.

        :param size: The number of configuration instances to generate.
        :param exhaustive: If True, generate all possible configurations.
        """
        hp_samples = None
        if exhaustive and (self.cs.estimate_size() != np.inf):
            try:
                hp_samples = generate_grid(self.cs)
            except ValueError:
                pass
        if hp_samples is None:
            hp_samples = self.cs.sample_configuration(size)
            if not isinstance(hp_samples, List):
                hp_samples = [hp_samples]
        return hp_samples

    @abstractmethod
    def build(self, config: Dict) -> Configuration:
        """
        Build the configuration with hyperparameters space.
        """
        pass


