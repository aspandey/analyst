from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

import datasource.app_db_weaviate as ds
import datasource.query_optimizer as qo

llm = ChatOllama(
    model="llama3.2:latest",
    # model = "deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.4,
    reasoning=False
    )


async def get_contextual_data(query: str) -> HumanMessage:
    print("Fetching context from Vector DB for query:", query)
    context = await ds.get_context_from_vector_db(query)
    human_message = HumanMessage(content="\n".join(context))
    return human_message

async def chat_with_user(user_query: str) -> str:

    system_message = SystemMessage(content= "You are an expert financial data analyst and extraction agent." \
        "Your primary task is to precisely analyze and extract structured information from raw financial data according to a user's request." \
        "  Core Directives:  " \
        "- Accuracy is paramount: Ensure all extracted values directly correspond to the source data." \
        "- Never hallucinate: If a piece of information is not present or cannot be determined from the provided context, " \
        "you must explicitly state that the information is unknown. Do not attempt to infer or guess." \
        "Data Schema and Glossary:" \
        "The data you receive will contain a set of key-value pairs. Use the following definitions to correctly interpret and extract the data:" \
        "- `company_or_stock_name`: The full name of the company or its stock symbol." \
        "- `industry_sector`: The broader economic sector or industry classification." \
        "- `quantity_of_shares`: The total number of shares held (numerical)." \
        "- `market_value_lacs_inr`: The total value of the holdings in lakh rupees (numerical)." \
        "- `portfolio_management_services_name`: The name of the portfolio management service who bought or sold the stocks." \
        "- `asset_under_managment_percentage`: The percentage of total assets under management (numerical)." \
        "- `data_month`: The calendar month for which the data is current." \
        "  Final Output:  " \
        "Based on the Context provided, try to answer user's query and return the requested information. " \
        "The output should be concise and directly address the user's query with proper human redable formatting." \
    )
    # RAG Cycle:
    #  1. Query Optimization: Refine the user's query to enhance relevance and precision for database searching.
    #  2. Context Retrieval: Fetch pertinent data from the vector database using the optimized query.
    #  3. Response Generation: Combine the retrieved context with the original user query to generate a comprehensive answer.
    opt_user_query = await qo.query_optimizer(user_query)
    # print(f"\n ============= Optimized Query for Vector DB: {opt_user_query} ================ \n")
    human_message = await get_contextual_data(opt_user_query)

    human_message.content += f"\n\nUser's Query: {user_query}\n Answer:"

    response = llm.invoke([system_message, human_message])
    return response.content

async def main():
    while True:
        print("================================== New Query =================================")
        user_input = input("Enter your query (type 'exit' to quit): ")

        if user_input.lower() == "exit":
            print("Exiting chat.")
            break

        response = await chat_with_user(user_input)
        print(f"\n +++++++++++++++++++++ AI Message +++++++++++++++++++++ \n {response} \n ")

import asyncio

if __name__ == "__main__":
    asyncio.run(main())    