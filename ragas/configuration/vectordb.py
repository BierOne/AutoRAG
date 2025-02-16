from .base import *
from .zoo import supported_vectordbs
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
        "tenant": DEFAULT_TENANT,
        "database": DEFAULT_DATABASE,
        "path": "/home/lyb/RAG/AutoRAG/experiments/db_resources/eli5",
    },
    "couchbase": {
        "index_name": "my_vector_index",  # replace your index name
        "host": "localhost",
        "port": 8091,
        "username": "Administrator",
        "password": "password",
        "bucket_name": "default", # replace your bucket name
        "scope_name": "_default",  # replace your scope name
        "connection_string": "couchbase://localhost",  # replace your connection string
        "text_key": "text",
        "embedding_key": "embedding",
        "scoped_index": True
    },
    "milvus": {
        "index_type": "IVF_FLAT", #
        "uri": "http://localhost:19530",
        "db_name": "", # replace your bucket name
        "token": "", # Set this if your Milvus server requires token-based authentication.
        "user": "", # Set this if your Milvus server requires username/password authentication.
        "password": "", #  Set this if your Milvus server requires username/password authentication.
        "timeout": 30.0, #  Specifies the timeout duration (in seconds) for Milvus operations.
    },
    "pinecone": {
        "index_name": "my_vector_index", # Sets the name of the Pinecone index where the vectors will be stored.
        "api_key": "your_api_key", # The API key for authentication with the Pinecone.
        "dimension": 1536, # The dimension of the vectors. This should correspond to the dimension of the embeddings generated by the specified embedding model.
        "cloud": "aws", # The cloud provider where the Pinecone index will be created. https://docs.pinecone.io/guides/indexes/understanding-indexes#serverless-indexes
        "region": "us-east-1", # The region where the Pinecone index will be hosted.
        "deletion_protection": "disabled", # Specifies whether deletion protection is enabled for the Pinecone index.
        "namespace": "default", # Specifies the namespace where the Pinecone index is located.
    },
    "qdrant": {
        "client_type": "docker", # available: [docker, cloud], The type of client to use for connecting to the Qdrant server.
        "url": "http://localhost:6333", # The URL of the Qdrant server.
        "host": "", # The host of the Qdrant server.
        "api_key": "", # The API key for the authentication of Qdrant server.
        "dimension": 1536, # The dimension of the vectors. This should correspond to the dimension of the embeddings generated by the specified embedding model.
        "parallel": 1, # Determines the number of parallel requests to the Qdrant server.
        "max_retries": 3, # Determines the maximum number of retries for the Qdrant server.
    },
    "weaviate": {
        "client_type": "docker", # available: [docker, cloud], The type of client to use for connecting to the Weaviate server.
        "host": "localhost", # The host of the Weaviate server.
        "port": 8080, # The port of the Weaviate server.
        "grpc_port": 50051, # The gRPC port of the Weaviate server.
        "url": None, # The URL of the Weaviate server.
        "api_key": None, # The API key for the authentication of Weaviate server.
        "text_key": "content", # Specifies the name of the property in Weaviate where the text data is stored.
    }
}


class VectorDBConfiguration(BaseConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build(config)


    def build(self, config: Dict) -> Configuration:
        assert "method" in config, "method is required in the configuration."
        used_methods = config.pop("method", [])
        method_weights = config.pop("method_weights", [1]*len(used_methods))

        cs = ConfigurationSpace(
            name="vectordb",
            space={
                "db_type": CategoricalHyperparameter("db_type", used_methods,
                                                           weights=method_weights),
            }
        )

        # Add method specific hyperparameters
        for method in supported_vectordbs:
            method_params = config.pop(method, {})
            # add hyperparameter configuration space
            if method in used_methods:
                cs.add_configuration_space(
                    prefix="[{}]".format(method),
                    delimiter="",
                    configuration_space=ConfigurationSpace(method_params),
                    parent_hyperparameter={"parent": cs["db_type"], "value": method},
                )

        # Add general hyperparameters
        if len(config) > 0:
            params = list(parse_hyperparameters_from_dict(config))
            cs.add(params)

        # # Add general hyperparameters
        # embedding_model = CategoricalHyperparameter("embedding_model", choices=["openai", "bert", "gpt-3"])
        # embedding_batch = UniformIntegerHyperparameter("embedding_batch", lower=50, upper=200, default_value=100)
        # similarity_metric = CategoricalHyperparameter("similarity_metric", choices=["cosine", "euclidean", "ip"],
        #                                               default_value="cosine")
        # collection_name = CategoricalHyperparameter("collection_name",
        #                                             choices=["collection1", "collection2", "collection"])
        # ingest_batch = UniformIntegerHyperparameter("ingest_batch", lower=50, upper=200, default_value=100)
        # cs.add([embedding_model, embedding_batch, similarity_metric, collection_name, ingest_batch])

        # add conditions
        conditions = []
        if "pinecone" in cs["db_type"].choices:
            # USE cs["collection_name"] only if vectordb_name != "pinecone"
            conditions.append(NotEqualsCondition(cs["collection_name"], cs["db_type"], "pinecone"))
        if "couchbase" in cs["db_type"].choices:
            # cs["similarity_metric"] == "ip" if vectordb_name == "couchbase"
            conditions.append(ForbiddenAndConjunction(
                ForbiddenEqualsClause(cs["db_type"], "couchbase"),
                ForbiddenInClause(cs['similarity_metric'], ["cosine", "euclidean"])
            ))
        # USE cs["ingest_batch"]  if vectordb_name in ["couchbase", "milvus", "pinecone"]
        # if any(db in cs["db_type"].choices for db in ["couchbase", "milvus", "pinecone"]):
        #     conditions.append(InCondition(cs["ingest_batch"], cs["db_type"], ["couchbase", "milvus", "pinecone"]))
        cs.add(conditions)

        return cs

    @staticmethod
    def load_static_params(module_type: str) -> Dict:
        """
        Get the default parameters for the module.
        These would not be used for hyperparameter optimization.
        """
        return VECTORDB_STATIC_PARAMS.get(module_type, {})


    def create_node_lines(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        """
        {
            'name': 'chroma_mpnet',
            'db_type': 'chroma',
            'client_type': 'persistent',
            'collection_name': 'huggingface_all_mpnet_base_v2',
            'embedding_model': 'huggingface_all_mpnet_base_v2',
        }
        """
        if samples is None:
            samples = self.sampling(size, **kwargs)
        node_lines = []
        for hp_config in samples:
            hp_config_dict = parse_hyperparameters_samples(dict(hp_config))
            module_type = hp_config_dict["db_type"]
            static_params = self.load_static_params(module_type)
            hp_config_dict.update(static_params)
            node_lines.append({
                "name": "chroma_large",
                **hp_config_dict
            })
        return node_lines

