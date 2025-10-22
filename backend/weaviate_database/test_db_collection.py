from weaviate_database.db_collection import AppWeaviateClient, WeaviateCollection
import data_process.parse_xlsx_sheet as pe
import data_process.data_preprocessing as data

def db_test():
    db_config  = {"host": "127.0.0.1", "port": 80, "grpc_port": 50051}
    COLLECTION_NAME = "StocksInfo"
    with AppWeaviateClient(**db_config) as cl:
        col = WeaviateCollection(client=cl)
        while True:
            print("What do you want to do:")
            print("1 - Create collection and insert objects")
            print("2 - Delete collection")
            print("3 - Retrieve objects for a query")
            print("4 - List all collections")
            print("5 - Get Collection config")
            print(f"6 - Fetch objects from collection:")
            print("7 - Exit the program")
            print("---------------------------------------------------")
            action = input("Enter Action (1/2/3/4/5/6/7): ").strip()
            
            if action == "1":
                COLLECTION_NAME = input("Enter collection name (default 'StocksInfo'): ").strip() or "StocksInfo"
                if COLLECTION_NAME in col.list_collection:
                    print(f"Collection '{COLLECTION_NAME}' already exists. Please choose a different name or delete the existing collection first.")
                    continue
                col.create_collection(COLLECTION_NAME)
                stocks_objects = pe.get_stock_info_from_xlsx(pe.STOCK_INFO_PATH)
                processed_data = data.data_preprocess_stock(stocks_objects)
                col.insert_objects_into_collection(COLLECTION_NAME, stocks_objects=processed_data)
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
                response = col.retrieve_objects_for_query(COLLECTION_NAME, user_query.lower(), target_vector="company_info")
                if not response or not response.objects:
                    print("No results found.")
                    continue
                count = 0
                for obj in response.objects:
                    print(f"\n {obj} \n")
                    count += 1
                    print(f"count = {count}")

            elif action == "4":
                collections = col.list_collection
                print("Collections in Weaviate:")
                for c in collections:
                    print(f"- {c}")

            elif action == "5":
                print(f"Collection config {col.get_config()}")
            elif action == "6":
                objects_num = input("How many objects you want to fetch (default 5): ").strip() or "5"
                try:
                    num = int(objects_num)
                except ValueError:
                    print("Invalid number. Please enter a valid integer.")
                    continue
                col.list_objects_from_collection(objects_num=num)
            elif action == "7":
                print("Exiting the program.")
                break
            else:
                print("Invalid action. Please enter 'create', 'delete', or 'retrieve'.")

if __name__ == "__main__":
    db_test()
