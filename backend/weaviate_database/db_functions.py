import weaviate

from weaviate.classes.config import Configure
from weaviate.client import WeaviateClient
from weaviate.outputs.query import QueryReturn
import weaviate.classes.query as wq
from weaviate.classes.query import HybridFusion
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

PROPERTIES_LIST  = [
    "company_or_stock_name",
    "industry_sector",
    "portfolio_management_services_name",
    "data_month",
    # "combined_text",
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

def create_filter_conditions(filters: dict) -> list:
    """
    Creates filter conditions based on the provided filters dictionary.
    
    Args:
        filters (dict): Dictionary where keys are property names and values are dictionaries containing
                        'values' (list of values to filter on) and 'operator' (AND/OR).
    
    Returns:
        list: List of filter conditions to be combined.
    """
    filter_conditions = []
    query_filters = filters
    # query_filters = filters.get("filters", {})
    for key, filter_info in query_filters.items():
        print(f"Processing filter for key: {key} with info: {filter_info}")
        dbg.info(f"Processing filter for key: {key} with info: {filter_info}")
        if key not in FILTER_PROP_NAMES:
            dbg.info(f"Property '{key}' is not in FILTER_PROP_NAMES. Skipping.")
            continue
            
        values = filter_info.get('values', [])
        operator = filter_info.get('operator', 'AND').upper()
        
        if not values or len(values) == 0:
            continue
            
        dbg.info(f"Applying filter on property '{key}' with values: {values} and operator: {operator}")
        print(f"Applying filter on property '{key}' with values: {values} and operator: {operator}")
        
        if len(values) == 1:
            # Single value: create simple equality filter
            filter_conditions.append(wq.Filter.by_property(key.lower()).equal(values[0].lower()))
        else:
            # Multiple values: use operator logic
            if operator == 'OR':
                # OR condition for multiple values of the same property
                or_filters = [wq.Filter.by_property(key.lower()).equal(val.lower()) for val in values]
                filter_conditions.append(wq.Filter.any_of(or_filters))
            else:
                # AND condition for multiple values of the same property
                and_filters = [wq.Filter.by_property(key.lower()).equal(val.lower()) for val in values]
                filter_conditions.append(wq.Filter.all_of(and_filters))
    
    return filter_conditions

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
    def list_collections(self) -> list:
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
        if collection_name in self.list_collections:
            print(f"Collection '{collection_name}' already exists.")
            return

        vector_config = create_vector_config(vector_names=VECTOR_NAMES)
        prop_config = create_properties_config(properties_list=PROPERTIES_LIST)

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

        if collection_name not in self.list_collections:
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
            filter_conditions = create_filter_conditions(filters)
            # query_filters=wq.Filter.all_of(filter_conditions) if filter_conditions else None
            
            response = collection.query.hybrid(
                query=user_query,
                query_properties=VECTOR_NAMES,
                max_vector_distance=0.6,
                alpha=0.00,
                limit=5000,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                target_vector=target_vector,
                # filters=query_filters,
                filters=wq.Filter.any_of(filter_conditions) if filter_conditions else None,
                return_metadata=wq.MetadataQuery(score=True, explain_score=True, certainty=True),
                return_properties=PROPERTIES_LIST,
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
            return_properties=PROPERTIES_LIST,
        )
        
        if not response or not response.objects:
            print("No objects found in the collection.")
            return
        
        for obj in response.objects:
            print(f"\n {obj.properties.get('company_or_stock_name')}")

        print(f"Total {len(response.objects)} objects retrieved from collection '{COLLECTION_NAME}'")

