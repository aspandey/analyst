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

def query_rewriter(user_input: str) -> str:
    system_message = SystemMessage(content=
        "You are a Query Rewriter for a vector search engine. "
        "The vector database has the following schema: "
        "`company_or_stock_name`, `industry_sector`, `data_month`, `portfolio_management_services_name`. "
        "Rephrase the user's natural language query into a short, concise set of keywords or phrases "
        "that directly map to these fields and can be used for keyword search. "
        "Avoid conversational words, filler text, or long sentences. "
        "Output should be a single line with keywords/phrases separated by commas. "
        "Example: 'Show holdings of HDFC Bank in July managed by Helios PMS' → "
        "'HDFC Bank, July, Helios PMS'."
    )


    human_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message, human_message])
    return response.content

def query_expander(user_input: str) -> str:
    system_message = SystemMessage(content=
        "You are a Query Expander for a vector search engine. "
        "Expand the user's query by adding relevant keywords, synonyms, and related phrases "
        "to improve semantic recall without changing its intent. "
        "Keep the output short and focused — a comma-separated list of meaningful terms only. "
        "Example: 'companies in finance sector' → 'finance companies, financial institutions, banking sector, finance industry'."
    )

    human_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message, human_message])
    return response.content

def query_decomposer(user_input: str) -> str:
    system_message = SystemMessage(content=
        "You are a Query Decomposer. Break complex user queries into smaller, focused sub-queries. "
        "Each sub-query should represent a single intent or a distinct piece of information that can be searched independently. "
        "Keep each sub-query short, clear, and in logical order. "
        "Example: 'Show me all companies in the finance sector and their total quantity in July' → "
        "1. companies in finance sector. "
        "2. total quantity of companies in July."
    )

    human_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message, human_message])
    return response.content

qr_input = "Can you tell me which companies are in the automobile sector for July?"
qr_output = "companies in the automobile sector during July"

qe_input = "finance sector companies"
qe_output = "finance companies, banking firms, financial institutions, NBFCs, fintech companies"

qd_input = "List companies in the finance sector and show their market value in August."
# 1. Companies in the finance sector
# 2. Market value of those companies in August

if __name__ == "__main__":
    print(" ===============  Query Rewriter Example  ===============   \n")
    res = query_rewriter(qr_input)
    print("Rewritten Query:", res)

    print("\n  ===============   Query Expander Example:  ===============   \n")
    
    res = query_expander(qe_input)
    print("Expanded Query:", res)

    print("\n  ===============   Query Decomposer Example:  ===============  \n")
    res = query_decomposer(qd_input)
    print("Decomposed Sub-Queries:", res)