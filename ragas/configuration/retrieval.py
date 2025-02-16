from .base import *
from .zoo import supported_retrievers
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, Mapping, Sequence, overload

import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, InCondition, NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant
from chromadb import (
	DEFAULT_TENANT,
	DEFAULT_DATABASE,
)

RETRIEVAL_STATIC_PARAMS = {
    # 'strategy': {'metrics': ['retrieval_f1',
    #        'retrieval_recall',
    #        'retrieval_precision'],
    "vectordb":{
        "vectordb": "default_vectordb",
    },
    "hybrid_cc":{
        "target_modules": "('bm25', 'vectordb')"
    },
    "hybrid_rrf":{
        "target_modules": "('bm25', 'vectordb')"
    },
}


class RetrievalConfiguration(BaseConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)



    def build(self, config: Dict) -> Configuration:
        assert "method" in config, "method is required in the configuration."
        used_methods = config.pop("method", [])
        method_weights = config.pop("method_weights", [1]*len(used_methods))

        # init configuration space
        cs = ConfigurationSpace(
            name="retrieval", # node_line_name in AutoRAG
            space={"module_type": CategoricalHyperparameter("module_type", used_methods, weights=method_weights),}
        )

        # Add method specific hyperparameters
        for method in supported_retrievers:
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
        return RETRIEVAL_STATIC_PARAMS.get(module_type, {})


    def create_node_lines(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        if samples is None:
            samples = self.sampling(size, **kwargs)
        node_lines = []
        for hp_config in samples:
            hp_config_dict = parse_hyperparameters_samples(dict(hp_config))
            module_type = hp_config_dict["module_type"]
            static_params = self.load_static_params(module_type)
            hp_config_dict.update(static_params)
            if module_type.startswith("hybrid"):
                top_k = hp_config_dict.get("top_k", 5)
                hp_config_dict.update({
                    "target_module_params": [
                        {
                            "top_k": top_k,
                        },
                        {
                            "top_k": top_k,
                            "vectordb": "chroma_large"
                        }
                    ],
                })
            node_line = {
                "node_line_name": "retrieve_node_line",
                "nodes": [{
                    "strategy": {
                        "metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]
                    },
                    "node_type": hp_config.config_space.name,
                    "modules": [hp_config_dict],
                }]
            }

            # if module_type.startswith("hybrid"):
                # node_line["nodes"][0]["modules"].extend([
                #     {'module_type': 'bm25', "top_k": top_k},
                #     {'module_type': 'vectordb', "vectordb": "chroma_large", "top_k": top_k}
                # ])
            node_lines.append(node_line)

        return node_lines

