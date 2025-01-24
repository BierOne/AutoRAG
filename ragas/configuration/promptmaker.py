from .base import *
# import ragas
import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant
from .zoo import prompt_maker_method


class PostRetrievalConfiguration(BaseConfiguration):
    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build()

    def post_build(self, cs: ConfigurationSpace):
        """
        Add general hyperparameters to the configuration space.
        """
        return cs


class PromptMakerConfiguration(PostRetrievalConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)

    def build(self, config) -> ConfigurationSpace|None:
        prompt_maker = config.get("prompt_maker")
        if prompt_maker is not None:
            method = []
            prompt_set = []
            try:
                for iter in prompt_maker["method"]:
                    if iter in prompt_maker_method:
                        method.append(iter)
                for k,v in prompt_maker["prompt"].items():
                    prompt_set.append(v)
            except:
                raise ValueError(
				f"Prompt Maker elements are missing."
			)

            cs = ConfigurationSpace(
                space={
                    "prompt_maker_methods": CategoricalHyperparameter("methods", choices=method),
                    "init_prompt": CategoricalHyperparameter("init_prompt", choices=prompt_set),
                }
            )

            return cs
        else:
            return None

    def sampling(self, size: Optional[int] = 1) -> Union[Configuration, List[Configuration]]:
        return self.cs.sample_configuration(size)



