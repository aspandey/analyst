from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

import json
import asyncio
    
llm = ChatOllama(
    model="llama3.2:latest",
    base_url="http://localhost:11434",
    temperature=0.4
    )

async def query_optimizer(user_input: str) -> str:
    system_message = SystemMessage(content= "You are an expert query optimizer designed to assist a vector database search engine." \
                            "Your sole purpose is to rephrase a user's natural language question into a set of highly relevant," \
                            "structured keywords and phrases." \
                            "Instructions:" \
                            "1.  Analyze the User Query: Carefully read the user's question to understand their core intent and the specific pieces of information they are looking for." \
                            "2.  Identify Relevant Schema Keys:** Map the user's request to the most relevant keys from the schema provided below." \
                            "3.  Generate Keywords: Create a concise, information-dense query using key terms from the user's input and the schema. Avoid conversational fillers like 'tell me about' or 'what is'." \
                            "4.  Format the Output: Your final output must be a clean, unstructured text string ready for vector embedding." \
                            "5.  DO NOT try to answer to the user query. ONLY optimize the query." \
                            "Database Schema for Reference:" \
                            "- `company_or_stock_name`: Should be the company name for which query is asked." \
                            "- `industry_sector`: Should be the sector name for which query is asked." \
                            "- `quantity_of_shares`: The total number of shares held (numerical)." \
                            "- `market_value_lacs_inr`: The total value of the holdings in lakh rupees (numerical)." \
                            "- `portfolio_management_services_name`: Should be the name of pms or portfolio_management_services for which query is asked" \
                            "- `asset_under_managment_percentage`: The percentage of total assets under management (numerical)." \
                            "- `data_month`: should be the month for which query was asked." \
                            "Final Output:" \
                            "The output should in json formted string with only the relevant keys and values that directly correspond to the user's query." \
                            "Example:" \
                            "User Input: Tell me the quantity of shares of Reliance Industries, which works in petrolium sector, held by Axis pms in the month of july?" \
                            'Your Output: {"company_or_stock_name": "Reliance Industries", "industry_sector": "petrolium", "data_month": "july", "portfolio_management_services_name": "Axis"}' \
    )
    human_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message, human_message])
    return response.content


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

heading = "company_or_stock_name	industry_sector	 quantity_of_shares 	 market_value_lacs_inr 	 asset_under_managment_percentage 	 data_month 	 portfolio_management_services_name "
# values = "Fortis Healthcare Ltd.	Healthcare Services	  56,842 	  487.39 	  2.53 	 july 	 Helios "
values = "HDFC Bank Limited	Banks	5,126,070.00	103,454.34	6.99	 July 	 Axis "
# Split the heading and values into lists, stripping extra whitespace
keys = [k.strip() for k in heading.split('\t') if k.strip()]
vals = [v.strip() for v in values.split('\t') if v.strip()]

# Create a dictionary from keys and values
data_dict = dict(zip(keys, vals))

# Convert the dictionary to a JSON string
json_string = json.dumps(data_dict)

user_query = "Tell me in which sector HDFC Bank Limited works and how many shares of it are held by Axis pms in the month of July and august?"

async def main():
    #  response = await data_optimizer(json_string)
     response = await query_optimizer(user_query)
     print(f"Optimized Query: {response}")
if __name__ == "__main__":
    asyncio.run(main())
    # user_query = "How many shares of Reliance Industries are held by the ABC fund in the month of May?"
    # optimized_query = asyncio.run(query_optimizer(user_query))
    # print(f"Optimized Query: {optimized_query}")

