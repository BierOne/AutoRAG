# from ragas.configuration.promptmaker import PromptMakerConfiguration
# import yaml

# # 假设你的YAML文件名为 'config.yaml'
# with open('./ragas/configuration/config.yaml', 'r') as file:
#     config = yaml.safe_load(file)

# # 打印读取的字典
# print(config)


# pm = PromptMakerConfiguration(config)
# print(pm.cs)
# print(pm.sampling(2))

# ls = {"llm": "adsad"}
# print(ls.get("m"))



from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, NotEqualsCondition, Categorical
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant


cs = ConfigurationSpace(
    space={
        "name": Categorical("name", ["gpt3.5", "mistralai_7b", "mistralai_7b_openailike","mistralai_7b_vllm","gpt4","Qwen"], weights=[1, 1,1,1,1,1]),
    }
)

cs.add_configuration_space(
    prefix="gpt3.5",
    delimiter="_",
    configuration_space=ConfigurationSpace(
        space={
            "framework": Categorical("framework", ["llama_index_llm"]),
            "llm": Categorical("llm", ["openai"]),
            "model": Categorical("model", ["gpt-3.5-turbo-16k"]),
            "max_token": Categorical("max_token", [512]),
            "batch": Categorical("batch", [16]),
            "temperature": UniformFloatHyperparameter("temperature", lower=0.0, upper=2.0),
        }
    ),
    parent_hyperparameter={"parent": cs["name"], "value": "gpt3.5"}
)
cs.add_configuration_space(
    prefix="mistralai_7b_huggingface",
    delimiter="_",
    configuration_space=ConfigurationSpace(
        space={
            "framework": Categorical("framework", ["llama_index_llm"]),
            "llm": Categorical("llm", ["huggingface"]),
            "model": Categorical("model", ["mistralai/Mistral-7B-Instruct-v0.2"]),
            "max_tokens": Categorical("max_tokens", [512]),
            "device_map": Categorical("device_map", ["auto"]),
            "model_kwargs": Categorical("model_kwargs", [{"torch_dtype": "float16"}]),
            "temperature": UniformFloatHyperparameter("temperature", lower=0.0, upper=2.0),
        }
    ),
    parent_hyperparameter={"parent": cs["name"], "value": "mistralai_7b"}
)
cs.add_configuration_space(
    prefix="mistralai_7b_openailike",
    delimiter="_",
    configuration_space=ConfigurationSpace(
        space={
            "framework": Categorical("framework", ["llama_index_llm"]),
            "llm": Categorical("llm", ["openailike"]),
            "model": Categorical("model", ["mistralai/Mistral-7B-Instruct-v0.2"]),
            "max_tokens": Categorical("max_tokens", [512]),
            "batch": Categorical("batch", [16]),
            "api_base": Categorical("api_base", ["your_api_base"]),
            "api_key": Categorical("api_key", ["your_api_key"]),
            "temperature": UniformFloatHyperparameter("temperature", lower=0.0, upper=2.0),
        }
    ),
    parent_hyperparameter={"parent": cs["name"], "value": "mistralai_7b"}
)
cs.add_configuration_space(
    prefix="mistralai_7b_vllm",
    delimiter="_",
    configuration_space=ConfigurationSpace(
        space={
            "framework": Categorical("framework", ["vllm"]),
            "llm": Categorical("llm", ["mistralai/Mistral-7B-Instruct-v0.2"]),
            "max_tokens": Categorical("max_tokens", [512]),
            "temperature": UniformFloatHyperparameter("temperature", lower=0.0, upper=2.0),
            "top_p": UniformFloatHyperparameter("top_p", lower=0.0, upper=1.0),
        }
    ),
    parent_hyperparameter={"parent": cs["name"], "value": "mistralai_7b"}
)
cs.add_configuration_space(
    prefix="gpt4",
    delimiter="_",
    configuration_space=ConfigurationSpace(
        space={
            "framework": Categorical("framework", ["openai_llm"]),
            "llm": Categorical("llm", ["gpt-4-turbo-2024-04-09"]),
            "max_tokens": Categorical("max_tokens", [512]),
            "batch": Categorical("batch", [16]),
            "api_key": Categorical("api_key", ["your_api_key"]),
            "truncate": Categorical("truncate", [4000]),
            "temperature": UniformFloatHyperparameter("temperature", lower=0.0, upper=2.0),
        }
    ),
    parent_hyperparameter={"parent": cs["name"], "value": "gpt4"}
)
cs.add_configuration_space(
    prefix="Qwen",
    delimiter="_",
    configuration_space=ConfigurationSpace(
        space={
            "framework": Categorical("framework", ["vllm_api"]),
            "uri": Categorical("uri", ["http://localhost:8012"]),
            "llm": Categorical("llm", ["Qwen/Qwen2.5-14B-Instruct-AWQ"]),
            "max_tokens": Categorical("max_tokens", [400]),
            "temperature": UniformFloatHyperparameter("temperature", lower=0.0, upper=2.0),
        }
    ),
    parent_hyperparameter={"parent": cs["name"], "value": "Qwen"}
)
cs.to_yaml("generator_config.yaml")
cs = ConfigurationSpace.from_yaml("generator_config.yaml")
print(cs)