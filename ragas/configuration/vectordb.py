from .base import *
import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant





class VectorDBConfiguration(BaseConfiguration):

    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)
        self.cs = self.build()

    def build(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(
            space={
                "vectordb_name": CategoricalHyperparameter("vectordb_name", ["chroma", "couchbase", "milvus",
                                                                                      "pinecone", "qdrant", "weaviate"],
                                                                    default="chroma", weights=[1, 1, 1, 1, 1, 1]),
            }
        )

        # Add general hyperparameters
        embedding_model = CategoricalHyperparameter("embedding_model", choices=["openai", "bert", "gpt-3"])
        embedding_batch = UniformIntegerHyperparameter("embedding_batch", lower=50, upper=200, default_value=100)
        similarity_metric = CategoricalHyperparameter("similarity_metric", choices=["cosine", "euclidean", "ip"],
                                                      default_value="cosine")
        collection_name = CategoricalHyperparameter("collection_name", choices=["collection1", "collection2", "collection"])
        cs.add_hyperparameters([embedding_model, embedding_batch, similarity_metric, collection_name])
        # add conditions
        cs.add([
            # USE cs["collection_name"] only if vectordb_name != "pinecone"
            NotEqualsCondition(cs["collection_name"], cs["vectordb_name"], "pinecone"),
            # cs["similarity_metric"] == "ip" if vectordb_name == "couchbase"
            ForbiddenAndConjunction(
                ForbiddenEqualsClause("vectordb_name", "couchbase"),
                ForbiddenInClause(cs['similarity_metric'], ["cosine", "euclidean"])
            )
        ])

        # add hyperparameter configuration space for chroma
        cs.add_configuration_space(
            prefix="[chroma]",
            delimiter="",  # No delimiter for the prefix
            configuration_space=ConfigurationSpace({
                "client_type": CategoricalHyperparameter("client_type", choices=["ephemeral", "persistent", "http", "cloud"]),
                "path": CategoricalHyperparameter("path", choices=["/path/to/db1", "/path/to/db2"]),
                "host": CategoricalHyperparameter("host", choices=["localhost", "remotehost"]),
                "port": Constant("port", value=8000),
                "ssl": CategoricalHyperparameter("ssl", choices=[True, False], default_value=False),
                "tenant": CategoricalHyperparameter("tenant", choices=["tenant1", "tenant2"]),
                "database": CategoricalHyperparameter("database", choices=["db1", "db2"]),
            }),
            parent_hyperparameter={"parent": cs["vectordb_name"], "value": "chroma"},
        )

        # add hyperparameter configuration space for couchbase
        cs.add_configuration_space(
            prefix="[couchbase]",
            delimiter="",
            configuration_space=ConfigurationSpace({
                "bucket_name": CategoricalHyperparameter("bucket_name", choices=["bucket1", "bucket2"]),
                "scope_name": CategoricalHyperparameter("scope_name", choices=["scope1", "scope2"]),
                "index_name": CategoricalHyperparameter("index_name", choices=["index1", "index2"]),
                "connection_string": CategoricalHyperparameter("connection_string", choices=["conn1", "conn2"]),
                "username": CategoricalHyperparameter("username", choices=["user1", "user2"]),
                "password": CategoricalHyperparameter("password", choices=["pass1", "pass2"]),
                "ingest_batch": UniformIntegerHyperparameter("ingest_batch", lower=50, upper=200, default_value=100),
                "text_key": CategoricalHyperparameter("text_key", choices=["text1", "text2"]),
                "embedding_key": CategoricalHyperparameter("embedding_key", choices=["embedding1", "embedding2"]),
                "scoped_index": CategoricalHyperparameter("scoped_index", choices=[True, False], default_value=True),
            }),
            parent_hyperparameter={"parent": cs["vectordb_name"], "value": "couchbase"},
        )

        # add hyperparameter configuration space for milvus
        cs.add_configuration_space(
            prefix="[milvus]",
            delimiter="",
            configuration_space=ConfigurationSpace({
                "index_type": CategoricalHyperparameter("index_type", choices=["IVF_FLAT", "IVF_SQ8"]),
                "uri": CategoricalHyperparameter("uri", choices=["http://localhost:19530", "http://remotehost:19530"]),
                "db_name": CategoricalHyperparameter("db_name", choices=["db1", "db2"]),
                "token": CategoricalHyperparameter("token", choices=["token1", "token2"]),
                "user": CategoricalHyperparameter("user", choices=["user1", "user2"]),
                "password": CategoricalHyperparameter("password", choices=["pass1", "pass2"]),
                "timeout": UniformFloatHyperparameter("timeout", lower=0.1, upper=10.0, default_value=1.0),
            }),
            parent_hyperparameter={"parent": cs["vectordb_name"], "value": "milvus"},
        )

        # add hyperparameter configuration space for pinecone
        cs.add_configuration_space(
            prefix="[pinecone]",
            delimiter="",
            configuration_space=ConfigurationSpace({
                "index_name": CategoricalHyperparameter("index_name", choices=["index1", "index2"]),
                "dimension": UniformIntegerHyperparameter("dimension", lower=128, upper=2048, default_value=1536),
                "cloud": CategoricalHyperparameter("cloud", choices=["aws", "gcp"]),
                "region": CategoricalHyperparameter("region", choices=["us-east-1", "us-west-2"]),
                "api_key": CategoricalHyperparameter("api_key", choices=["key1", "key2"]),
                "deletion_protection": CategoricalHyperparameter("deletion_protection", choices=["enabled", "disabled"], default_value="disabled"),
                "namespace": CategoricalHyperparameter("namespace", choices=["default", "namespace1"]),
                "ingest_batch": UniformIntegerHyperparameter("ingest_batch", lower=50, upper=500, default_value=200),
            }),
            parent_hyperparameter={"parent": cs["vectordb_name"], "value": "pinecone"},
        )

        # add hyperparameter configuration space for qdrant
        cs.add_configuration_space(
            prefix="[qdrant]",
            delimiter="",
            configuration_space=ConfigurationSpace({
                "client_type": CategoricalHyperparameter("client_type", choices=["docker", "cloud"]),
                "url": CategoricalHyperparameter("url", choices=["http://localhost:6333", "http://remotehost:6333"]),
                "host": CategoricalHyperparameter("host", choices=["localhost", "remotehost"]),
                "api_key": CategoricalHyperparameter("api_key", choices=["key1", "key2"]),
                "dimension": UniformIntegerHyperparameter("dimension", lower=128, upper=2048, default_value=1536),
                "ingest_batch": UniformIntegerHyperparameter("ingest_batch", lower=50, upper=200, default_value=64),
                "parallel": UniformIntegerHyperparameter("parallel", lower=1, upper=10, default_value=1),
                "max_retries": UniformIntegerHyperparameter("max_retries", lower=1, upper=10, default_value=3),
            }),
            parent_hyperparameter={"parent": cs["vectordb_name"], "value": "qdrant"},
        )

        # add hyperparameter configuration space for weaviate
        cs.add_configuration_space(
            prefix="[weaviate]",
            delimiter="",
            configuration_space=ConfigurationSpace({
                "client_type": CategoricalHyperparameter("client_type", choices=["docker", "cloud"]),
                "host": CategoricalHyperparameter("host", choices=["localhost", "remotehost"]),
                "port": Constant("port", value=8080),
                "grpc_port": Constant("grpc_port", value=50051),
                "url": CategoricalHyperparameter("url", choices=["http://localhost:8080", "http://remotehost:8080"]),
                "api_key": CategoricalHyperparameter("api_key", choices=["key1", "key2"]),
                "text_key": CategoricalHyperparameter("text_key", choices=["content", "text"]),
            }),
            parent_hyperparameter={"parent": cs["vectordb_name"], "value": "weaviate"},
        )

        return cs

    def sampling(self, size: Optional[int] = 1) -> Union[Configuration, List[Configuration]]:
        return self.cs.sample_configuration(size)

