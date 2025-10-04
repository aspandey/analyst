from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

import json
import asyncio
    
llm = ChatOllama(
    model="llama3.2:latest",
    # model = "deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.2,
    reasoning=False
    )

async def query_optimizer(user_input: str) -> str:
    system_message = SystemMessage(content= "You are an expert query optimizer for a vector database search engine.\n" \
        "You are an expert query optimizer designed to assist a vector database search engine. \n"\
        "Your only job is to rephrase a user's natural language query into a clean, minimal, and relevant keyword-based form.\n"\
        "IMPORTANT RULES: \n"\
        # "1. NEVER assume or invent any value (for example, company names like 'HDFC Bank'). \n"\
        # "2. ONLY include values explicitly mentioned in the user input. \n"\
        # "3. If the user asks about companies, stocks, or PMS generally (without naming one), leave that property blank or exclude it. \n"\
        "1. Keep the output short, factual, and concise — avoid long text or filler words.\n "\
        "2. Focus on keywords and phrases that capture the core intent of the user's question.\n"\
        "3. Use terms directly from the user's input wherever possible.\n"\
        "4. Avoid conversational phrases like 'tell me about' or 'what is'.\n"\
        "5. DO NOT try to answer the user query. ONLY optimize the query.\n"\
        "6. Do not explain anything just give the optimized query.\n"\
        "7. DO NOT include your thinking process in your response.\n"\
        # "5. Return a single JSON object containing only the directly relevant keys and their actual mentioned values. \n"\
        # "6. If a property value is not present in the query, DO NOT fabricate or guess it — just skip it.\n"\
        # "7. The output must be strictly JSON — no explanations, no natural language.\n"\
        # "8. DO NOT include any property that is not explicitly mentioned in the user query.\n"\
        # "9. DO NOT include properties with null or empty values.\n"\
        # "- company_or_stock_name\n" \
        # "- industry_sector\n" \
        # "- quantity_of_shares\n" \
        # "- market_value_lacs_inr\n"\
        # "- asset_under_managment_percentage\n"\
        # "- portfolio_management_services_name\n"\
        # "- data_month\n"\
        # "Example:\n"\
        # "User: Get total quantity of companies working in the retail industry.\n"\
        # 'Output: {"industry_sector": "retail"}'\
    )
    human_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message, human_message])
    print(f"\n ============== Optimized Query Response: {response.content} ================ \n\n")
    return response.content


# async def query_optimizer(user_input: str) -> str:
#     system_message = SystemMessage(content= "You are an expert query optimizer designed to assist a vector database search engine." \
#                             "Your sole purpose is to rephrase a user's natural language question into a set of highly relevant," \
#                             "structured keywords and phrases." \
#                             "Instructions:" \
#                             "1.  Analyze the User Query: Carefully read the user's question to understand their core intent and the specific pieces of information they are looking for." \
#                             "2.  Identify Relevant Schema Keys: Map the user's request to the most relevant keys from the schema provided below." \
#                             "3.  Generate Keywords: Create a concise, information-dense query using key terms from the user's input and the schema. Avoid conversational fillers like 'tell me about' or 'what is'." \
#                             "4.  Format the Output: Your final output must be a clean, unstructured text string ready for vector embedding." \
#                             "5.  DO NOT try to answer to the user query. ONLY optimize the query." \
#                             "Database Schema for Reference:" \
#                             "- `company_or_stock_name`: Should be the company name for which query is asked." \
#                             "- `industry_sector`: Should be the sector name for which query is asked." \
#                             "- `quantity_of_shares`: The total number of shares held (numerical)." \
#                             "- `market_value_lacs_inr`: The total value of the holdings in lakh rupees (numerical)." \
#                             "- `portfolio_management_services_name`: Should be the name of pms or portfolio_management_services for which query is asked" \
#                             "- `asset_under_managment_percentage`: The percentage of total assets under management (numerical)." \
#                             "- `data_month`: should be the month for which query was asked." \
#                             "Final Output:" \
#                             "should be concise and direct text string, without any additional commentary." \
#                             "The output should be a string with core intent and the specific pieces of information they are looking for" \
#                             "NO JSON FORMAT, only text string" \
#     )
#     human_message = HumanMessage(content=user_input)
#     response = llm.invoke([system_message, human_message])
#     return response.content

async def data_optimizer(user_input: str) -> str:
    system_message = SystemMessage(content= "You will receive json string, you need to convert it into a plain text string using keys and values " \
                            "- `company_or_stock_name`: The full name of the company or its stock symbol." \
                            "- `industry_sector`: The broader economic sector or industry classification." \
                            "- `quantity_of_shares`: The total number of shares held (numerical)." \
                            "- `market_value_lacs_inr`: The total value of the holdings in lakh rupees (numerical)." \
                            "- `portfolio_management_services_name`: The name of the portfolio management service." \
                            "- `asset_under_managment_percentage`: The percentage of total assets under management (numerical)." \
                            "- `data_month`: The calendar month for which the data is current." \
                            "**Final Output:**" \
                            "The output should be concise and direct, without any additional commentary." \
                            "Example: " \
                            'json_string = {"company_or_stock_name": "Fortis Healthcare Ltd.", "industry_sector": "Healthcare Services", "quantity_of_shares": "56,842", "market_value_lacs_inr": "487.39", "asset_under_managment_percentage": "2.53", "data_month": "july", "portfolio_management_services_name": "Helios"}' \
                            "plain text string = Fortis Healthcare Ltd. works in Healthcare Services sector and helios pms owns 56,842 number of shares of it which values INR 487.39 and 2.53 percentage of its asset_under_managment_percentage"
    )

    human_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message, human_message])
    return response.content

user_query = "Tell me in which sector HDFC Bank Limited works and how many shares of it are held by Axis pms in the month of July and august?"

async def main():
     response = await query_optimizer(user_query)
     print(f"Optimized Query: {response}")
if __name__ == "__main__":
    asyncio.run(main())

