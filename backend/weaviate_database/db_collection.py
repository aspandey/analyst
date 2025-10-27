import weaviate

from weaviate.classes.config import Configure
from weaviate.client import WeaviateClient
from weaviate.outputs.query import QueryReturn
import weaviate.classes.query as wq

from weaviate.classes.query import HybridFusion

from typing import Optional
from data_process.parse_xlsx_sheet import get_stock_info_from_xlsx
import data_process.parse_xlsx_sheet as pe
import data_process.data_preprocessing as data

import weaviate.classes.config as wc
from weaviate.classes.config import Configure, VectorDistances
from debug.logger_config import dbg

EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_API_URL = "http://host.docker.internal:11434" # ollama server url if calling from docker, example, calling from weaviate container
COLLECTION_NAME = "StocksInfo"

VECTOR_NAMES = [
    "company_or_stock_name",
    "industry_sector",
    "data_month",
    "portfolio_management_services_name",
    "combined_text"
]

FILTER_PROP_NAMES = [
    "company_or_stock_name",
    "industry_sector",
    "data_month",
    "portfolio_management_services_name",
]

SEARCHABLE_PROP_NAMES = ["combined_text"]

NUMERIC_PROP_NAMES = [
    "quantity_of_shares",
    "market_value_lacs_inr",
    "asset_under_managment_percentage"
]

properties_list  = [
    "company_or_stock_name",
    "industry_sector",
    "portfolio_management_services_name",
    "data_month",
    "combined_text",
    "quantity_of_shares",
    "market_value_lacs_inr",
    "asset_under_managment_percentage",
]

def create_vector_config(vector_names: list) -> list:
    """
    Creates and returns vector configuration for the collection.
    
    Returns:
        list: List of vector configurations for each named vector.
    """
    if not vector_names:
        raise ValueError("Vector names list cannot be empty.")

    vector_config = [
        wc.Configure.Vectors.text2vec_ollama(
            name=vec_name,
            source_properties=[vec_name],
            api_endpoint=OLLAMA_API_URL,
            model=EMBEDDING_MODEL,
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=128,
                max_connections=32,
                quantizer=Configure.VectorIndex.Quantizer.bq(),
                ef=64,
            ),
        )
        for vec_name in vector_names
    ]
    return vector_config


def create_properties_config(properties_list: list) -> list:
    """
    Creates and returns property configuration for the collection.
    
    Args:
        properties_list (list): List of property names to configure.
    
    Returns:
        list: List of property configurations.
    """
    if not properties_list:
        raise ValueError("Properties list cannot be empty.")
    
    property_configs = []
    
    for prop in properties_list:
        if prop in FILTER_PROP_NAMES:
            property_configs.append(
                wc.Property(name=prop, data_type=wc.DataType.TEXT, index_filterable=True)
            )
        elif prop in SEARCHABLE_PROP_NAMES:
            property_configs.append(
                wc.Property(name=prop, data_type=wc.DataType.TEXT, index_searchable=True)
            )
        elif prop in NUMERIC_PROP_NAMES:
            property_configs.append(
                wc.Property(
                    name=prop,
                    data_type=wc.DataType.NUMBER,
                    index_filterable=False,
                    index_searchable=False,
                    vectorize_property_name=False
                )
            )
    
    return property_configs

class AppWeaviateClient:
    def __init__(self, host: str = "localhost", port: int = 8080, grpc_port: int = 50051):
        if port <= 0 or grpc_port <= 0:
            raise ValueError("Port numbers must be positive integers")
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.client: Optional[WeaviateClient] = None

    def connect(self) -> WeaviateClient:
        """
        Connects to the Weaviate instance using the provided host, port, and grpc_port.
        Returns:
            WeaviateClient: The connected client instance.
        """
        self.client = weaviate.connect_to_local(
            host=self.host,
            port=self.port,
            grpc_port=self.grpc_port,
        )
        return self.client

    def close(self):
        """
        Closes the connection to the Weaviate instance.
        """
        if self.client:
            self.client.close()
            self.client = None

    def __enter__(self) -> WeaviateClient:
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
class WeaviateCollection:
    def __init__(self, client: WeaviateClient):
        self.client = client

    def get_config(self) -> None:
        """Prints the Weaviate meta configuration."""
        try:
            meta_info = self.client.get_meta()
            print(meta_info.get("modules", {}))
        finally:
            self.client.close()

    @property
    def list_collection(self) -> list:
        """Lists all collection names in the Weaviate instance."""
        names = []
        collections = self.client.collections.list_all()
        for collection in collections:
            names.append(collection)
        return names
    
    def create_collection(self, collection_name: str = "COLLECTION_NAME") -> None:
        """
        Creates a Weaviate collection with the specified configuration.
        Args:
            collection_name (str): Name of the collection to create.
        """
        if not self.client:
            raise ValueError("Weaviate client is not connected. Call connect() first.")
        if not collection_name:
            raise ValueError("Collection name cannot be empty.")
        if collection_name in self.list_collection:
            print(f"Collection '{collection_name}' already exists.")
            return

        vector_config = create_vector_config(vector_names=VECTOR_NAMES)
        prop_config = create_properties_config(properties_list=properties_list)

        self.client.collections.create(
            name=collection_name,
            properties=prop_config,
            vector_config=vector_config,
        )

    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes a Weaviate collection.
        Args:
            collection_name (str): Name of the collection to delete.
        """
        if not self.client:
            raise ValueError("Weaviate client is not connected. Call connect() first.")

        if not collection_name:
            raise ValueError("Collection name cannot be empty.")

        if collection_name not in self.list_collection:
            print(f"Collection '{collection_name}' does not exist.")
            return

        self.client.collections.delete(collection_name)
    
    def insert_objects_into_collection(self, collection_name: str, stocks_objects: list[dict]) -> None:
        """
        Inserts objects into a specified collection in batches.
        Args:
            collection_name (str): Name of the collection.
            stocks_objects (list[dict]): List of objects to insert.
        """
        if not self.client:
            raise ValueError("Weaviate client is not connected. Call connect() first.")
        if not collection_name:
            raise ValueError("Collection name cannot be empty.")
        if not stocks_objects:
            raise ValueError("Source objects cannot be empty.")

        collection = self.client.collections.get(collection_name)

        with collection.batch.fixed_size(batch_size=200) as batch:
            for src_obj in stocks_objects:
                batch.add_object(
                    properties={key: value for key, value in src_obj.items()}
                )
                if batch.number_errors > 10:
                    print("Batch import stopped due to excessive errors.")
                    break
            
        failed_objects = collection.batch.failed_objects

        if failed_objects:
            print(f"Number of failed imports: {len(failed_objects)}")
            print(f"First failed object: {failed_objects[0]}")

    def retrieve_objects_for_query(self, collection_name: str, user_query: str, filters: dict, target_vector: str = "combined_text") -> QueryReturn:
        """
        Queries and prints objects from a collection using a different type of search.
        Args:
            collection_name (str): Name of the collection to query.
            user_query (str): The query string to search for.
        """
        try:

            if not self.client:
                raise ValueError("Weaviate client is not connected. Call connect() first.")
            if not collection_name:
                raise ValueError("Collection name cannot be empty.")

            collection = self.client.collections.get(collection_name)


            # Build filter from filters dict
            filter_conditions = []
            for key, values in filters.items():
                # if values and len(values) > 0:
                #     if len(values) == 1:
                #         filter_conditions.append(wq.Filter.by_property(key).equal(values[0]))
                #     else:
                #         # OR condition for multiple values of same property
                #         or_filters = [wq.Filter.by_property(key).equal(val) for val in values]
                #         filter_conditions.append(wq.Filter.any_of(or_filters))

                if key in FILTER_PROP_NAMES and values and len(values) > 0:
                    dbg.info(f"Applying filter on property '{key}' with values: {values}")
                    if len(values) == 1:
                        filter_conditions.append(wq.Filter.by_property(key).equal(values[0]))
                        break

            # Combine all filter conditions with AND
            # final_filter = wq.Filter.any_of(filter_conditions) if filter_conditions else None
            # dbg.info(f"Final filter for query (type: {type(final_filter).__name__}): {final_filter}")

            response = collection.query.hybrid(
                query=user_query,
                query_properties=VECTOR_NAMES,
                max_vector_distance=0.6,
                alpha=0.25,
                limit=5000,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                target_vector=target_vector,
                filters=wq.Filter.all_of(filter_conditions) if filter_conditions else None,
                return_metadata=wq.MetadataQuery(score=True, explain_score=True, certainty=True),
                return_properties=["company_or_stock_name"],
            )

        except Exception as e:
            print(f"Error retrieving objects for query: {e}")
            response = None

        return response

    def list_objects_from_collection(self, objects_num: int = 5) -> None:
        """
        Lists a specified number of objects from the collection.
        Args:
            objects_num (int): Number of objects to list.
        """
        if not self.client:
            raise ValueError("Weaviate client is not connected. Call connect() first.")
        if objects_num <= 0:
            raise ValueError("Number of objects must be a positive integer.")
        
        collection = self.client.collections.get(COLLECTION_NAME)
        response = collection.query.fetch_objects(
            limit=objects_num,
            # include_vector=True,
            return_properties=properties_list,
        )
        
        if not response or not response.objects:
            print("No objects found in the collection.")
            return
        
        for obj in response.objects:
            print(f"\n {obj.properties.get('company_or_stock_name')}")

        print(f"Total {len(response.objects)} objects retrieved from collection '{COLLECTION_NAME}'")


def format_investment_summary(data_dict: dict[str, str]) -> str:
    """Return a short summary from the investment data dict."""
    try:
        combined_text = data_dict.get("company_or_stock_name", "")
        # dbg.info(f"Combined Text: {combined_text}")
        dbg.info(f"Formatting summary for company: {combined_text}")
        return combined_text

    except KeyError as e:
        return f"Error: Missing key in input data: {e}. Cannot generate summary."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

    
# async def get_context_from_vector_db(user_query_str: str) -> list[dict[str, str]]:
async def get_context_from_vector_db(user_query_str: str, query_filters: dict) -> list[str]:

    db_config  = {"host": "127.0.0.1", "port": 80, "grpc_port": 50051}
    COLLECTION_NAME = "StocksInfo"
    context_list = []
    count = 1
    select  = 1
    with AppWeaviateClient(**db_config) as cl:

        col = WeaviateCollection(client=cl)
        response = col.retrieve_objects_for_query(COLLECTION_NAME, user_query_str.lower(), filters=query_filters)

        if not response or not response.objects:
            return context_list

        for obj in response.objects:
            count += 1
            score = obj.metadata.score if obj.metadata and obj.metadata.score else 0.0
            if score < 0.3:
                continue
            
            select += 1
            stocks_str = format_investment_summary(obj.properties)
            context_list.append(stocks_str)
        
        print(f"Total {count} objects retrieved from Vector DB")
        print(f"Total {select} objects selected from Vector DB")

    return context_list
