from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

import datasource.app_db_weaviate as ds

llm = ChatOllama(
    model="llama3.2:latest",
    base_url="http://localhost:11434",
    temperature=0.4
    )

async def query_optimizer(user_input: str) -> str:
    system_message = SystemMessage(content= "You are an expert query optimizer designed to assist a vector database search engine." \
                            "Your sole purpose is to rephrase a user's natural language question into a set of highly relevant," \
                            "structured keywords and phrases." \
                            "**Instructions:**" \
                            "1.  **Analyze the User Query:** Carefully read the user's question to understand their core intent and the specific pieces of information they are looking for." \
                            "2.  **Identify Relevant Schema Keys:** Map the user's request to the most relevant keys from the schema provided below." \
                            "3.  **Generate Keywords:** Create a concise, information-dense query using key terms from the user's input and the schema. Avoid conversational fillers like 'tell me about' or 'what is'." \
                            "4.  **Format the Output:** Your final output must be a clean, unstructured text string ready for vector embedding." \
                            "5.  ** DO NOT try to answer to the user query. ONLY optimize the query. **" \
                            "**Database Schema for Reference:**" \
                            "- `company_or_stock_name`: The full name of the company or its stock symbol." \
                            "- `industry_sector`: The broader economic sector or industry classification." \
                            "- `quantity_of_shares`: The total number of shares held (numerical)." \
                            "- `market_value_lacs_inr`: The total value of the holdings in lakh rupees (numerical)." \
                            "- `portfolio_management_services_name`: The name of the portfolio management service." \
                            "- `asset_under_managment_percentage`: The percentage of total assets under management (numerical)." \
                            "- `data_month`: The calendar month for which the data is current." \
                            "**Final Output:**" \
                            "The output should be concise and direct, without any additional commentary."
                            # "**Example:**" \
                            # "* **User Input:** How many shares of Reliance Industries are held by the ABC fund in the month of May?" \
                            # "* **Your Output:** Shares quantity for Reliance Industries (`stock_name`) held by ABC fund (`pms_name`) in May (`month`)." \
    )
    human_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message, human_message])
    return response.content


async def chat_with_user(user_query: str) -> str:

    system_message = SystemMessage(content= "You are an expert financial data analyst and extraction agent." \
        "Your primary task is to precisely analyze and extract structured information from raw financial data according to a user's request." \
        "**Core Directives:**" \
        "- **Accuracy is paramount:** Ensure all extracted values directly correspond to the source data." \
        "- **Strictly adhere to the schema:** Only extract information that fits the provided definitions." \
        "- **Never hallucinate:** If a piece of information is not present or cannot be determined from the provided context, " \
        "you must explicitly state that the information is unknown. Do not attempt to infer or guess." \
        "**Data Schema and Glossary:**" \
        "The data you receive will contain a set of key-value pairs. Use the following definitions to correctly interpret and extract the data:" \
        "- `company_or_stock_name`: The full name of the company or its stock symbol." \
        "- `industry_sector`: The broader economic sector or industry classification." \
        "- `quantity_of_shares`: The total number of shares held (numerical)." \
        "- `market_value_lacs_inr`: The total value of the holdings in lakh rupees (numerical)." \
        "- `portfolio_management_services_name`: The name of the portfolio management service." \
        "- `asset_under_managment_percentage`: The percentage of total assets under management (numerical)." \
        "- `data_month`: The calendar month for which the data is current." \
        "**Final Output:**" \
        "Based on the user's query and the data, return the requested information. " \
        "The output should be concise and directly address the user's question with proper human redable formatting." \
    )
    opt_user_query = await query_optimizer(user_query)
    print("Optimized Query for Vector DB:", opt_user_query)
    human_message = await get_contextual_data(opt_user_query)
    human_message.content += f"\n\nUser's Query: {user_query}\nAnswer:"
    response = llm.invoke([system_message, human_message])
    return response.content

async def get_contextual_data(query: str) -> HumanMessage:
    context = await ds.get_context_from_vector_db(query)
    human_message = HumanMessage(content=str(context))
    return human_message

async def main():
    while True:
        user_input = input("Enter your query (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting chat.")
            break

        response = await chat_with_user(user_input)
        print("Response from Ollama:", response)

import asyncio

if __name__ == "__main__":
    asyncio.run(main())    