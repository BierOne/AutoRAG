from .base import *
from .promptmaker import PostRetrievalConfiguration
from .zoo import supported_generators,available_model_zoo
import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, \
    NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, Constant




GENERATOR_STATIC_PARAMS = {
    "LlamaIndexLLM": {
        "api_base": "your_api_base",
        "api_key": "your_api_key",
        "request_timeout": 120.0,
    },
    "OpenAILLM": {
        "truncate": True,
        # "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # "api_key": "sk-ab6eb49be7934c4f86678574618c646a", # OpenAI API key. You can also set this to env variable OPENAI_API_KEY.
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-e46fb251c74d4c64a4c2835333e994a3",
        # OpenAI API key. You can also set this to env variable OPENAI_API_KEY.
        "request_timeout": 1200.0,
    },
    "Vllm": {

    },
    "VllmAPI": {
        "uri": "http://localhost:8012", # The URI of the vLLM API server.
    }
}


class GeneratorConfiguration(PostRetrievalConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)

    def build(self, config) -> Dict | None:
        assert "method" in config, "method is required in the configuration."
        used_methods = config.pop("method", [])
        method_weights = config.pop("method_weights", [1]*len(used_methods))

        # init configuration space
        cs = ConfigurationSpace(
            name="generator", # node_type in AutoRAG
            space={"module_type": CategoricalHyperparameter("module_type", used_methods, weights=method_weights),}
        )

        # Add method specific hyperparameters
        for method in supported_generators:
            method_params = config.pop(method, {})
            # print(method, method_params)
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

        # add default conditions
        # TODO: Add conditions to check available models
        # cs.add([
        #     # USE cs["collection_name"] only if vectordb_name != "pinecone"
        #     NotEqualsCondition(cs["collection_name"], cs["vectordb_name"], "pinecone"),
        # ])

        return cs

    @staticmethod
    def load_static_params(module_type: str) -> Dict:
        """
        Get the default parameters for the module.
        These would not be used for hyperparameter optimization.
        """
        return GENERATOR_STATIC_PARAMS.get(module_type, {})

    def create_nodes(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        """
            {
                'node_type': 'generator',
                'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
                'modules': [
                    {
                        'module_type': 'llama_index_llm',
                        'llm': 'ollama',
                        'model': 'llama3',
                        'temperature': 0.5,
                        'batch': 1
                    }
                ]
            }
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


