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
        "1. Keep the output short, factual, and concise — avoid long text or filler words.\n "\
        "2. Focus on keywords and phrases that capture the core intent of the user's question.\n"\
        "3. Use terms directly from the user's input wherever possible.\n"\
        "4. Avoid conversational phrases like 'tell me about' or 'what is'.\n"\
        "5. DO NOT try to answer the user query. ONLY optimize the query.\n"\
        "6. Do not explain anything just give the optimized query.\n"\
        "7. DO NOT include your thinking process in your response.\n"\
    )
    human_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message, human_message])
    print(f"\n ============== Optimized Query Response: {response.content} ================ \n\n")
    return response.content


async def classify_named_vector(user_query: str) -> list[str]:
    system_message = SystemMessage(content= """
        You are a query router for a vector database with multiple named vectors:
        - company_vector: represents company or stock names
        - sector_vector: represents industry or sector names
        - pms_vector: represents portfolio management service, pms, names
        Decide which named vector(s) should be used for searching the following user query.
        Output one or more vector names as a JSON list, e.g. ["company_vector"] or ["sector_vector"] or ["company_vector", "sector_vector"].
        Only output the JSON list and nothing else.
        """)

    human_message = HumanMessage(content=user_query)
    response = llm.invoke([system_message, human_message])
    # print(f"\n ============== Named Vector Classification Raw Response: {response.content} ================ \n\n")
    return json.loads(response.content)

async def repharase_company_info (company_info: str) -> str:
    system_message = SystemMessage(content= """
        You are a data cleaner and transformer.
        Rephrase the following text to be more concise and clear, removing any redundant or unnecessary information.
        Ensure the core meaning and important details are preserved.
        Output the rephrased text only and nothing else.
        """)

    human_message = HumanMessage(content=company_info)
    response = llm.invoke([system_message, human_message])
    return response.content

# user_query = "Tell me in which sector HDFC Bank Limited works and how many shares of it are held by Axis pms in the month of July and august?"
# user_query = "list of companies working in finance sector"
# user_query = "Does Helios pms holds shares of ola electric?"
# user_query = "IN which sector helios pms invest most?"
user_query = "Give me the list of companies held by axis pms?"

company_info = "sun pharmaceutical industries limited is a company in the pharmaceuticals and biotechnology sector. \
               sun pharmaceutical industries limited is part of the axis PMS portfolio. In july, axis held 498078 shares \
               of sun pharmaceutical industries limited valued at market value 8500.7 lakh INR, representing 0.57 percent of total assets"

async def main():
    # response = await classify_named_vector(user_query)
    response = await repharase_company_info(company_info)
    print(f"\n ============== Named Vector Classification Response: {response} ================ )\n\n")

    #  response = await query_optimizer(user_query)
    #  print(f"Optimized Query: {response}")
if __name__ == "__main__":
    asyncio.run(main())

