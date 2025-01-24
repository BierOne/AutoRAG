from .base import *
# import ragas
import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, \
    NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, Constant
from .zoo import llm_framework


QUERY_EXPANSION_STATIC_PARAMS = {
}


class QueryExpansionConfiguration(BaseConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)
        self.para = self.translate(config)

    def build(self, config) -> Dict | None:
        cs = ConfigurationSpace(
            space={
                "query_expansion_name": CategoricalHyperparameter("query_expansion_name",
                                                            ["pass_query_expansion", "query_decompose", "hyde", "multi_query_expansion"],
                                                            weights=[1, 1, 1, 1]),
            }
        )
        # Add general hyperparameters
        prompt = CategoricalHyperparameter("prompt")
        cs.add([prompt])
        return cs


    def sampling(self, size: Optional[int] = 1) -> Union[Configuration, List[Configuration]]:
        return self.cs.sample_configuration(size)

