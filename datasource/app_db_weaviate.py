import weaviate
from weaviate.classes.config import Configure
from weaviate.client import WeaviateClient
from weaviate.outputs.query import QueryReturn
import weaviate.classes.query as wq

from typing import Optional
from datasource.parse_excel import get_stock_info_from_xlsx
import datasource.parse_excel as pe

import json
import weaviate.classes.config as wc
from weaviate.classes.config import Configure, VectorDistances

EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_API_URL = "http://host.docker.internal:11434" # ollama server url if calling from docker, example, calling from weaviate container
COLLECTION_NAME = "StocksInfo"
VECTOR_NAMES = [
    "company_or_stock_name",
    "industry_sector",
    "data_month",
    "portfolio_management_services_name"
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
stocks_objects = pe.get_stock_info_from_xlsx(pe.STOCK_INFO_PATH)

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
    # VECTOR_NAMES = ["stock_name", "industry", "pms_name", "month"]
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
            name="stock_name_industry_pms_month_vector",
            source_properties= VECTOR_NAMES,
            api_endpoint=OLLAMA_API_URL,
            model=EMBEDDING_MODEL,
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=128,
                max_connections=128,
                quantizer=Configure.VectorIndex.Quantizer.bq(),
                ef=-1,
                dynamic_ef_factor=15,
                dynamic_ef_min=200,
                dynamic_ef_max=1000,
            ),
            )
        ]

        self.client.collections.create(
            name=collection_name,
            properties=[
            # Enable keyword indexing on relevant text properties
            
            wc.Property(name="company_or_stock_name", data_type=wc.DataType.TEXT, index_filterable=True, index_searchable=True),
            wc.Property(name="industry_sector", data_type=wc.DataType.TEXT, index_filterable=True, index_searchable=True),
            wc.Property(name="quantity_of_shares", data_type=wc.DataType.NUMBER),
            wc.Property(name="market_value_lacs_inr", data_type=wc.DataType.NUMBER),
            wc.Property(name="asset_under_managment_percentage", data_type=wc.DataType.NUMBER),
            wc.Property(name="portfolio_management_services_name", data_type=wc.DataType.TEXT, index_filterable=True, index_searchable=True),
            wc.Property(name="data_month", data_type=wc.DataType.TEXT, index_filterable=True, index_searchable=True),
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

    def retrieve_objects_for_query(self, collection_name: str, user_query: str) -> QueryReturn:
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
                query_properties=VECTOR_NAMES,
                max_vector_distance=0.6,
                limit=100,
                alpha=0.5,
                target_vector="stock_name_industry_pms_month_vector",
                return_metadata=wq.MetadataQuery(score=True, explain_score=True),
                return_properties=properties_list,
            )

        except Exception as e:
            print(f"Error retrieving objects for query: {e}")
            response = None
        
        # print(f"Query response: {response}")
        return response

async def get_context_from_vector_db(user_query: str) -> list[dict[str, str]]:
    db_config  = {"host": "127.0.0.1", "port": 80, "grpc_port": 50051}
    COLLECTION_NAME = "StocksInfo"
    context_list = []
    total_count = 1
    select_count = 0
    with AppWeaviateClient(**db_config) as cl:
        col = WeaviateCollection(client=cl)
        user_query = user_query.strip()
        response = col.retrieve_objects_for_query(COLLECTION_NAME, user_query.lower())
        if not response or not response.objects:
            return context_list

        for obj in response.objects:
            total_count += 1
            # print(f"============= Total Count : {total_count} =========================\n")
            score = obj.metadata.score if obj.metadata and obj.metadata.score else 0.0
            if score < 0.3:
                continue
            
            select_count += 1
            print(f"============= Selected Count :  {select_count} =========================\n")
            # print(f"Hybrid score - explain: {obj.metadata.score:.3f} - {obj.metadata.explain_score}\n")
            # print(f"======================================\n")
            score = obj.metadata.score if obj.metadata and obj.metadata.score else 0.0
            obj_str = json.dumps(obj.properties)
            print(f"Object: {obj_str}\n")
            context_list.append(obj_str)
        
        print(f"============= Total: {total_count}   Selected :  {select_count} =========================\n")
        total_count = 1
        select_count = 0
        return context_list


# =============== TEST CODE ======================
# p3 -m datasource.app_db_weaviate
# ================================================
def db_test():
    db_config  = {"host": "127.0.0.1", "port": 80, "grpc_port": 50051}

    with AppWeaviateClient(**db_config) as cl:
        col = WeaviateCollection(client=cl)
        while True:
            print("What do you want to do:")
            print("1 - Create collection and insert objects")
            print("2 - Delete collection")
            print("3 - Retrieve objects for a query")
            print("4 - List all collections")
            print("5 - Get Collection config")
            print("6 - Exit the program")
            print("---------------------------------------------------")
            action = input("Enter Action (1/2/3/4/5/6): ").strip()
            COLLECTION_NAME = "StocksInfo"
            if action == "1":
                COLLECTION_NAME = input("Enter collection name (default 'StocksInfo'): ").strip() or "StocksInfo"
                col.create_collection(COLLECTION_NAME)
                col.insert_objects_into_collection(COLLECTION_NAME, stocks_objects)
                print(f"Collection '{COLLECTION_NAME}' created and objects inserted.")
            elif action == "2":
                COLLECTION_NAME = input("Enter collection name to delete : ").strip()
                if not COLLECTION_NAME:
                    print("Collection name cannot be empty.")
                    continue
                col.delete_collection(COLLECTION_NAME)
                print(f"Collection '{COLLECTION_NAME}' deleted (if it existed).")
            elif action == "3":
                COLLECTION_NAME = input("Enter collection name (default 'StocksInfo'): ").strip() or "StocksInfo"
                user_query = input("Enter your query: ").strip()
                response = col.retrieve_objects_for_query(COLLECTION_NAME, user_query.lower()
                )
                if not response or not response.objects:
                    print("No results found.")
                    continue
                
                for obj in response.objects:
                    print(f"\n {obj} \n")

            # print(obj.properties.get("description", "No title"))
            elif action == "4":
                collections = col.list_collection
                print("Collections in Weaviate:")
                for c in collections:
                    print(f"- {c}")
            elif action == "5":
                print(f"Collection config {col.get_config()}")
            elif action == "6":
                metainfo = cl.get_meta()
                print(json.dumps(metainfo, indent=2))
                print("Exiting the program.")
                break
            else:
                print("Invalid action. Please enter 'create', 'delete', or 'retrieve'.")

if __name__ == "__main__":
    db_test()
