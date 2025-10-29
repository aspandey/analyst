import weaviate
from weaviate.client import WeaviateClient
from weaviate_database.db_functions import WeaviateCollection

from typing import Optional
from debug.logger_config import dbg

EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_API_URL = "http://host.docker.internal:11434" # ollama server url if calling from docker, example, calling from weaviate container
COLLECTION_NAME = "StocksInfo"

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


def format_investment_summary(data_dict: dict[str, str]) -> str:
    """Return a short summary from the investment data dict."""
    try:
        # Define the desired order of fields
        field_order = [
            'company_or_stock_name',
            'portfolio_management_services_name',
            'data_month',
            'industry_sector',
            'quantity_of_shares',
            'market_value_lacs_inr',
            'asset_under_managment_percentage'
        ]
        
        formatted_parts = []
        
        # Process fields in the specified order
        for key in field_order:
            if key not in data_dict:
                continue
            
            value = data_dict[key]
            
            # Skip empty values
            if not value:
                continue
            
            # Format key: replace underscores with spaces and title case
            formatted_key = key.replace('_', ' ').title()
            
            # Format value: strip and capitalize if it's a string
            formatted_value = str(value).strip()
            if formatted_value and not formatted_value[0].isupper() and formatted_value[0].isalpha():
                formatted_value = formatted_value.capitalize()
            
            formatted_parts.append(f"{formatted_key}: {formatted_value}")
        
        # Add any remaining fields not in the specified order
        for key, value in sorted(data_dict.items()):
            if key in field_order:
                continue
            
            # Skip empty values
            if not value:
                continue
            
            # Format key: replace underscores with spaces and title case
            formatted_key = key.replace('_', ' ').title()
            
            # Format value: strip and capitalize if it's a string
            formatted_value = str(value).strip()
            if formatted_value and not formatted_value[0].isupper() and formatted_value[0].isalpha():
                formatted_value = formatted_value.capitalize()
            
            formatted_parts.append(f"{formatted_key}: {formatted_value}")
        
        # Use newline for better readability and consistency
        combined_text = "\n".join(formatted_parts)
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
            print(f"Object Properties: {obj.properties}")
            score = obj.metadata.score if obj.metadata and obj.metadata.score else 0.0
            # if score < 0.3:
            #     continue
            
            select += 1
            stocks_str = format_investment_summary(obj.properties) + "\n"
            context_list.append(stocks_str)
        
        print(f"Total {count} objects retrieved from Vector DB")
        print(f"Total {select} objects selected from Vector DB")

    return context_list
