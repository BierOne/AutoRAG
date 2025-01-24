from .base import *
# import ragas
import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, \
    NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, Constant
from .zoo import llm_framework


GENERATOR_STATIC_PARAMS = {
    "LlamaIndexLLM": {
        "api_base": "your_api_base",
        "api_key": "your_api_key",
        "llm_available_models":{ # TODO: available models from official package
            "openai": ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106"],
            "ollama": ["llama3"],
            "openailike": ["mistralai/Mistral-7B-Instruct-v0.2"],
        }
    },
    "OpenAILLM": {
        "truncate": True,
        "api_key": "your_api_key", # OpenAI API key. You can also set this to env variable OPENAI_API_KEY.
    },
    "Vllm": {

    },
    "VllmAPI": {
        "uri": "http://localhost:8012", # The URI of the vLLM API server.
    }
}


class GeneratorConfiguration(BaseConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)
        self.para = self.translate(config)

    def build(self, config) -> Dict | None:
        cs = ConfigurationSpace(
            space={
                "generator_name": CategoricalHyperparameter("generator_name",
                                                            ["LlamaIndexLLM", "OpenAILLM", "Vllm", "VllmAPI"],
                                                            weights=[1, 1, 1, 1]),
            }
        )
        # Add general hyperparameters
        temperature = CategoricalHyperparameter("temperature", choices=[0.5, 1.0, 1.5])
        batch = CategoricalHyperparameter("batch", choices=[16, 32])
        max_token = Constant("max_token", 512)
        cs.add([temperature, batch, max_token])
        return cs

        # generator = config.get("generator")
        # llm_group = {}
        # if generator is not None:
        #     try:
        #         for k,v in generator.items():
        #             if v["framwwork"] in llm_framework:
        #                 if v.get("top_k") is not None:
        #                     llm_group[v["name"]] = ConfigurationSpace(
        #                         space={
        #                             "temperature": UniformFloatHyperparameter("temperature", lower=v["temperature"][0], upper=v["temperature"][1]),
        #                             "top_k": UniformFloatHyperparameter("top_k", lower=v["top_k"][0], upper=v["top_k"][1])
        #                         }
        #                     )
        #                 else:
        #                     llm_group[v["name"]] = ConfigurationSpace(
        #                         space={
        #                             "temperature": UniformFloatHyperparameter("temperature", lower=v["temperature"][0], upper=v["temperature"][1]),
        #                         }
        #         )
        #     except:
        #         raise ValueError(

    # 		f"framework in generator is necessary. Or some elements in generator are missing."
    # 	)
    # return llm_group

    def sampling(self, size: Optional[int] = 1) -> Union[Configuration, List[Configuration]]:
        return self.cs.sample_configuration(size)

    def translate(self, config) -> Dict:
        return self.cs.sample_configuration(size)
