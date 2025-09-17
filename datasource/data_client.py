import requests

# Ask users input query and then send it to data server to get the response.
# data server is using embedded and generative models to get the answers.
# data app (FastAPI) is running on http://127.0.0.1:5000

DATA_SERVER_HOST = "http://127.0.0.1"
DATA_SERVER_PORT = "5000"
DATA_SERVER_URL = f"{DATA_SERVER_HOST}:{DATA_SERVER_PORT}"


def get_stock_analysis():
    while True:
        user_query = input("Enter your query: ")
        if user_query.strip().lower() == "bye":
            print("Goodbye!")
            break
        response = requests.post(
            f"{DATA_SERVER_URL}/query",
            json={
                "query": user_query
            }
        )
        print("==========================")
        print(f"User Query:\n{user_query}")
        print("==========================")
        print(f"Response :\n{response.json()["answer"]}")
        print("==========================")

def main():
    print("Welcome to the Stock Analysis System!!!")
    print("Type 'bye' to exit.")
    get_stock_analysis()

if __name__ == "__main__":
    main()