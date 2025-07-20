import os

from google.cloud import aiplatform, storage
from langchain import hub
from llama_index.core import (
    Document,
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import ObjectIndex
from llama_index.core.prompts import LangchainPromptTemplate
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore


def create_vector_search_index(
    index_name: str, index_dimensions: int
) -> aiplatform.MatchingEngineIndex:
    """
    Creates a Vector Index
    NOTE : This operation can take upto 30 minutes
    """

    # check if index exists
    index_names = [
        index.resource_name
        for index in aiplatform.MatchingEngineIndex.list(
            filter=f"display_name={index_name}"
        )
    ]

    if len(index_names) == 0:
        print(f"Creating Vector Search index {index_name} ...")
        vs_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=index_name,
            dimensions=index_dimensions,
            # distance_measure_type="DOT_PRODUCT_DISTANCE",
            shard_size="SHARD_SIZE_SMALL",
            index_update_method="STREAM_UPDATE",  # allowed values BATCH_UPDATE , STREAM_UPDATE,
            approximate_neighbors_count=3,
        )
        print(
            f"Vector Search index {vs_index.display_name} created with resource name {vs_index.resource_name}"
        )
    else:
        vs_index = aiplatform.MatchingEngineIndex(index_name=index_names[0])
        print(
            f"Vector Search index {vs_index.display_name} exists with resource name {vs_index.resource_name}"
        )

    return vs_index


def create_vector_search_endpoint(
    endpoint_name: str,
) -> aiplatform.MatchingEngineIndexEndpoint:
    """
    Creates a Vector Search endpoint.
    """
    endpoint_names = [
        endpoint.resource_name
        for endpoint in aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f"display_name={endpoint_name}"
        )
    ]

    if len(endpoint_names) == 0:
        print(f"Creating Vector Search index endpoint {endpoint_name} ...")
        vs_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=endpoint_name, public_endpoint_enabled=True
        )
        print(
            f"Vector Search index endpoint {vs_endpoint.display_name} created with resource name {vs_endpoint.resource_name}"
        )
    else:
        vs_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_names[0]
        )
        print(
            f"Vector Search index endpoint {vs_endpoint.display_name} exists with resource name {vs_endpoint.resource_name}"
        )

    return vs_endpoint


def deploy_vector_search_endpoint(
    vs_index: aiplatform.MatchingEngineIndex,
    vs_endpoint: aiplatform.MatchingEngineIndexEndpoint,
    index_name: str,
) -> aiplatform.MatchingEngineIndexEndpoint:
    """
    Deploys a Vector Search endpoint.
    """
    # check if endpoint exists
    index_endpoints = [
        (deployed_index.index_endpoint, deployed_index.deployed_index_id)
        for deployed_index in vs_index.deployed_indexes
    ]

    if len(index_endpoints) == 0:
        print(
            f"Deploying Vector Search index {vs_index.display_name} at endpoint {vs_endpoint.display_name} ..."
        )
        vs_deployed_index = vs_endpoint.deploy_index(
            index=vs_index,
            deployed_index_id=index_name,
            display_name=index_name,
            machine_type="e2-standard-16",
            min_replica_count=1,
            max_replica_count=1,
        )
        print(
            f"Vector Search index {vs_index.display_name} is deployed at endpoint {vs_deployed_index.display_name}"
        )
    else:
        vs_deployed_index = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=index_endpoints[0][0]
        )
        print(
            f"Vector Search index {vs_index.display_name} is already deployed at endpoint {vs_deployed_index.display_name}"
        )

    return vs_deployed_index
     