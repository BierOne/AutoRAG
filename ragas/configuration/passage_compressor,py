from .base import *



GENERATOR_STATIC_PARAMS = {
    "tree_summarize": None, # must start with a llm in generator
    "refine": None, # must start with a llm in generator
    "longllmlingua":{
        "model_name": "NousResearch/Llama-2-7b-hf",
        "instructions": "Given the context, please answer the final question", # Optional instructions for the LLM
        "target_token": 300, # The target token count for the output, default to 300.
    }




}


class PassageCompressorConfiguration(BaseConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)
        self.para = self.translate(config)

    def build(self, config=None) -> ConfigurationSpace | None:
        cs = ConfigurationSpace(
            space={
                "compressor_name": CategoricalHyperparameter("compressor_name",
                                                            ["tree_summarize", "refine", "longllmlingua"],
                                                            weights=[1, 1, 1]),
            }
        )
        return cs

    def sampling(self, size: Optional[int] = 1) -> Union[Configuration, List[Configuration]]:
        return self.cs.sample_configuration(size)

    def translate(self, config) -> Dict:
        return self.cs.sample_configuration(size)
