from abc import abstractmethod
from typing import List, Tuple, Union
from .util import load_yaml_config
import os
from typing import Optional, Dict
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, \
    NotEqualsCondition, Configuration, InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, Constant


class BaseConfiguration:

    def __init__(
        self,
        config: Dict,
        project_dir: Optional[str] = None
    ):
        self.config = config
        self.project_dir = os.getcwd() if project_dir is None else project_dir

    @abstractmethod
    def sampling(self, size: Optional[int] = 1) -> Configuration | list[Configuration]:
        """
        Sample the configuration.

        :param size: The number of configuration instances to generate.
        """
        pass

    @abstractmethod
    def build(self, config: Dict) -> None:
        """
        Build the configuration with hyperparameters space.
        """
        pass

    @classmethod
    def generate_config(cls, yaml_path: str, project_dir: Optional[str] = None):
        """
        generate component config to load autorag runner.
        """
        config = load_yaml_config(yaml_path)
        return cls(config, project_dir=project_dir)

