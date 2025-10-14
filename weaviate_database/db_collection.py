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

properties_list  = [
    "company_or_stock_name",
    "industry_sector",
    "quantity_of_shares",
    "market_value_lacs_inr",
    "asset_under_managment_percentage",
    "data_month",
    "portfolio_management_services_name"
]

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
        vector_config = [
            wc.Configure.Vectors.text2vec_ollama(
                name="company_info",
                # source_properties= VECTOR_NAMES,
                source_properties= ["combined_text"],
                api_endpoint=OLLAMA_API_URL,
                model=EMBEDDING_MODEL,
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    ef_construction=128,
                    max_connections=32,
                    quantizer=Configure.VectorIndex.Quantizer.bq(),
                    ef=64,
                ),
            ),
        ]
        self.client.collections.create(
            name=collection_name,
            properties=[
                # Enable keyword indexing (inverted index) on relevant text properties
                wc.Property(name="company_or_stock_name", data_type=wc.DataType.TEXT, index_filterable=True),
                wc.Property(name="industry_sector", data_type=wc.DataType.TEXT, index_filterable=True),
                wc.Property(name="portfolio_management_services_name", data_type=wc.DataType.TEXT, index_filterable=True),
                wc.Property(name="data_month", data_type=wc.DataType.TEXT, index_filterable=True),

                wc.Property(name="combined_text", data_type=wc.DataType.TEXT, index_searchable=True),
                
                wc.Property(name="quantity_of_shares", data_type=wc.DataType.NUMBER, index_filterable=False, index_searchable=False, vectorize_property_name=False),
                wc.Property(name="market_value_lacs_inr", data_type=wc.DataType.NUMBER, index_filterable=False, index_searchable=False, vectorize_property_name=False),
                wc.Property(name="asset_under_managment_percentage", data_type=wc.DataType.NUMBER, index_filterable=False, index_searchable=False, vectorize_property_name=False),
            ],
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

    def retrieve_objects_for_query(self, collection_name: str, user_query: str, target_vector: str = "company_info") -> QueryReturn:
        """
        Queries and prints objects from a collection using a near-text search.
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
            response = collection.query.hybrid(
                query=user_query,
                # vector= vector of the query to be used for vector seracg
                query_properties=VECTOR_NAMES,
                max_vector_distance=0.4,
                alpha=0.7,
                # limit=5000,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                auto_limit=True,
                target_vector=target_vector,
                filters=None,
                return_metadata=wq.MetadataQuery(score=True, explain_score=True, certainty=True),
                return_properties=["combined_text"],
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
            print(f"\n {obj.properties.get("company_or_stock_name")}")
        print(f"Total {len(response.objects)} objects retrieved from collection '{COLLECTION_NAME}'")

def format_investment_summary(data_dict: dict[str, str]) -> str:
    """
    Converts a dictionary containing financial data into a human-readable summary string.

    Args:
        data_dict: A dictionary containing specific keys like 'company_or_stock_name',
                   'industry_sector', 'data_month', etc.

    Returns:
        A formatted string summarizing the investment data.
    """
    try:
        # Extract the values from the dictionary, converting names to title case for readability
        # company = data_dict["company_or_stock_name"].title()
        # sector = data_dict["industry_sector"]
        # month = data_dict["data_month"]
        # pms_name = data_dict["portfolio_management_services_name"].title()
        # aum_percentage = data_dict["asset_under_managment_percentage"]
        # shares_quantity = data_dict["quantity_of_shares"]
        # market_value = data_dict["market_value_lacs_inr"]
        combined_text = data_dict.get("combined_text", "")
        # Use an f-string for clear, concise string construction
        # summary_string = (
        #     f"{company} works in {sector} sector "
        #     f"month: {month}, pms: {pms_name},  aum percentage: {aum_percentage},"
        #     f"quantity of shares: {shares_quantity}, value in INR: {market_value}"
        # )

        # return summary_string
        return combined_text

    except KeyError as e:
        return f"Error: Missing key in input data: {e}. Cannot generate summary."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

    
# async def get_context_from_vector_db(user_query_str: str) -> list[dict[str, str]]:
async def get_context_from_vector_db(user_query_str: str) -> list[str]:
    db_config  = {"host": "127.0.0.1", "port": 80, "grpc_port": 50051}
    COLLECTION_NAME = "StocksInfo"
    context_list = []
    count = 1
    select  = 1
    with AppWeaviateClient(**db_config) as cl:
        col = WeaviateCollection(client=cl)
        response = col.retrieve_objects_for_query(COLLECTION_NAME, user_query_str.lower())
        if not response or not response.objects:
            return context_list
        for obj in response.objects:
            count += 1
            score = obj.metadata.score if obj.metadata and obj.metadata.score else 0.0
            if score < 0.3:
                continue
            
            select += 1
            stocks_str = format_investment_summary(obj.properties)
            # print(f"Object from Vector DB: {stocks_str} \n")
            context_list.append(stocks_str)
        
        print(f"Total {count} objects retrieved from Vector DB")
        print(f"Total {select} objects selected from Vector DB")
    return context_list
