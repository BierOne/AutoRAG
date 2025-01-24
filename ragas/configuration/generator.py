from .base import *
# import ragas
import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant
from .zoo import llm_framework



class GeneratorConfiguration(BaseConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)
        self.para = self.translate(config)

    def build(self, config) -> Dict|None:
        generator = config.get("generator")
        if generator is not None:
            name = []
            llm_group = {}
            try:
                for k,v in generator.items():
                    if v["framwwork"] in llm_framework:
                        if v.get("top_k") is not None:
                            llm_group[v["name"]] = ConfigurationSpace(
                                space={
                                    "temperature": UniformFloatHyperparameter("temperature", lower=v["temperature"][0], upper=v["temperature"][1]),
                                    "top_k": UniformFloatHyperparameter("top_k", lower=v["top_k"][0], upper=v["top_k"][1])
                                }
                            )
                        else:
                            llm_group[v["name"]] = ConfigurationSpace(
                                space={
                                    "temperature": UniformFloatHyperparameter("temperature", lower=v["temperature"][0], upper=v["temperature"][1]),
                                }
                )
            except:
                raise ValueError(
				f"framework in generator is necessary. Or some elements in generator are missing."
			)


            return llm_group
        else:
            return None

    def sampling(self, size: Optional[int] = 1) -> Union[Configuration, List[Configuration]]:
        return self.cs.sample_configuration(size)

    def translate(self, config) -> Dict:
            
            return self.cs.sample_configuration(size)


