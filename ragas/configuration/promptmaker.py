from .base import *
from .zoo import supported_prompt_makers

import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant


PROMPT_MAKER_STATIC_PARAMS = {}



class PostRetrievalConfiguration(BaseConfiguration):

    @abstractmethod
    def post_build(self, cs: ConfigurationSpace) -> None:
        """
        Add generator hyperparameters to the configuration space.

        :param cs: The configuration space to which hyperparameters will be added.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def create_nodes(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        """
        Create nodes from the hyperparameters.

        :param size: The number of nodes to create.
        :param samples: The hyperparameters to use.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def create_node_lines(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> List[Dict]:
        nodes = self.create_nodes(size, samples, **kwargs)
        node_lines = [{
            "node_line_name": "post_retrieve_node_line",
            "nodes": [node]
        } for node in nodes]
        return node_lines



class PromptMakerConfiguration(PostRetrievalConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)

    def build(self, config: Dict) -> Configuration:
        assert "method" in config, "method is required in the configuration."
        used_methods = config.pop("method", [])
        method_weights = config.pop("method_weights", [1]*len(used_methods))

        cs = ConfigurationSpace(
            name="prompt_maker",  # node_type
            space={
                "module_type": CategoricalHyperparameter("module_type", used_methods,
                                                           weights=method_weights),
            }
        )

        # Add method specific hyperparameters
        for method in supported_prompt_makers:
            method_params = config.pop(method, {})
            # add hyperparameter configuration space
            if method in used_methods:
                cs.add_configuration_space(
                    prefix="[{}]".format(method),
                    delimiter="",
                    configuration_space=ConfigurationSpace(method_params),
                    parent_hyperparameter={"parent": cs["module_type"], "value": method},
                )

        # Add general hyperparameters
        if len(config) > 0:
            params = list(parse_hyperparameters_from_dict(config))
            cs.add(params)

        return cs


    @staticmethod
    def load_static_params(module_type: str) -> Dict:
        """
        Get the default parameters for the module.
        These would not be used for hyperparameter optimization.
        """
        return PROMPT_MAKER_STATIC_PARAMS.get(module_type, {})



    def create_nodes(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        """
        As default, each node only contain one prompt_maker module,
        So the evaluation of Prompt Maker will be skipped in AutoRAG Search.
        In this way, we do not need to set the generator_modules in the Prompt Maker Params.
        {
            'node_type': 'prompt_maker',
            'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
            'modules': [
                {
                    'module_type': 'fstring',
                    'prompt': 'Read the passages and answer the given question... '
                }
            ]
        },
        """
        if samples is None:
            samples = self.sampling(size, **kwargs)
        nodes = []
        for hp_config in samples:
            hp_config_dict = parse_hyperparameters_samples(dict(hp_config))
            module_type = hp_config_dict["module_type"]
            static_params = self.load_static_params(module_type)
            hp_config_dict.update(static_params)

            nodes.append({
                "node_type": hp_config.config_space.name,
                'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
                "modules": [hp_config_dict],
            })
        return nodes





