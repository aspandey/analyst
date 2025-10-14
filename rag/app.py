from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

import weaviate_database.db_collection as ds
import query_optimizer.query_transformer as qo
from prompts.chat_prompt import CHAT_SYSTEM_PROMPT

chat_response_llm = ChatOllama(
    model="llama3.2:latest",
    # model = "deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.4,
    reasoning=False
    )

# async def get_contextual_data(query: str) -> HumanMessage:
#     print("Fetching context from Vector DB for query:", query)
#     context = await ds.get_context_from_vector_db(query)
#     human_message = HumanMessage(content="\n".join(context))
#     return human_message

async def chat_with_user(user_query: str) -> str:
    """
    Handles a user chat interaction using a Retrieval-Augmented Generation (RAG) workflow.

    Args:
        user_query (str): The user's input query.

    Returns:
        str: The generated response based on retrieved context and the user's query.

    Workflow:
        1. Optimizes the user's query for improved relevance in database searches.
        2. Retrieves contextual data from a vector database using the optimized query.
        3. Generates a comprehensive answer by combining the retrieved context with the original user query.
    """
    system_message = SystemMessage(content=CHAT_SYSTEM_PROMPT)

    opt_user_query = await qo.query_optimizer(user_query)
    # human_message = await get_contextual_data(opt_user_query)
    context = await ds.get_context_from_vector_db(opt_user_query)
    human_message = HumanMessage(content="\n".join(context))

    human_message.content += f"\n\nUser's Query: {user_query}\n Answer:"
    # response = chat_response_llm.invoke([system_message, human_message])
    response = chat_response_llm.ainvoke([system_message, human_message])
    return response.content

############# Test code for chat_with_user ############# 
# p3 -m rag.app

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