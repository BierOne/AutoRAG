from .base import *
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, Mapping, Sequence, overload

import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, InCondition, NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant
from chromadb import (
	DEFAULT_TENANT,
	DEFAULT_DATABASE,
)

VECTORDB_STATIC_PARAMS = {
    "chroma": {
        "client_type": "persistent",
        "host": "localhost",
        "port": 8000,
        "ssl": False,
        "headers": None,
        "api_key": None,
        "tanant": DEFAULT_TENANT,
        "database": DEFAULT_DATABASE,
    },


}

def get_db_static_params(method: str) -> Dict:
    """
    Get the default parameters for the database.
    These would not be used for hyperparameter optimization.
    """
    pass

class VectorDBConfiguration(BaseConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build()


    def build(self, config: Dict) -> None:
        cs = ConfigurationSpace(
            space={
                "vectordb_name": CategoricalHyperparameter("vectordb_name", ["chroma", "couchbase", "milvus",
                                                                             "pinecone", "qdrant", "weaviate"],
                                                           weights=[1, 1, 1, 1, 1, 1]),
            }
        )

        # Add general hyperparameters
        embedding_model = CategoricalHyperparameter("embedding_model", choices=["openai", "bert", "gpt-3"])
        embedding_batch = UniformIntegerHyperparameter("embedding_batch", lower=50, upper=200, default_value=100)
        similarity_metric = CategoricalHyperparameter("similarity_metric", choices=["cosine", "euclidean", "ip"],
                                                      default_value="cosine")
        collection_name = CategoricalHyperparameter("collection_name",
                                                    choices=["collection1", "collection2", "collection"])
        ingest_batch = UniformIntegerHyperparameter("ingest_batch", lower=50, upper=200, default_value=100)
        cs.add([embedding_model, embedding_batch, similarity_metric, collection_name, ingest_batch])

        # add conditions
        cs.add([
            # USE cs["collection_name"] only if vectordb_name != "pinecone"
            NotEqualsCondition(cs["collection_name"], cs["vectordb_name"], "pinecone"),
            # cs["similarity_metric"] == "ip" if vectordb_name == "couchbase"
            ForbiddenAndConjunction(
                ForbiddenEqualsClause(cs["vectordb_name"], "couchbase"),
                ForbiddenInClause(cs['similarity_metric'], ["cosine", "euclidean"])
            ),
            # USE cs["ingest_batch"]  if vectordb_name in ["couchbase", "milvus", "pinecone"]
            InCondition(cs["ingest_batch"], cs["vectordb_name"], ["couchbase", "milvus", "pinecone"]),
        ])

        # add hyperparameter configuration space for milvus
        cs.add_configuration_space(
            prefix="",
            delimiter="",
            configuration_space=ConfigurationSpace({
                "index_type": CategoricalHyperparameter("index_type", choices=["IVF_FLAT", "IVF_SQ8"]),
            }),
            parent_hyperparameter={"parent": cs["vectordb_name"], "value": "milvus"},
        )
        return cs

    def sampling(self, size: Optional[int] = 1) -> Union[Configuration, List[Configuration]]:
        return self.cs.sample_configuration(size)

