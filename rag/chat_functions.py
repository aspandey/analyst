from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

import weaviate_database.db_collection as ds
import query_optimizer.query_transformer as qo
from prompts.chat_prompt import FINANCE_EXPERT_SYSTEM_PROMPTS
from debug.logger_config import dbg

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
    system_message = SystemMessage(content=FINANCE_EXPERT_SYSTEM_PROMPTS["V2"])

    opt_user_query = await qo.query_optimizer(user_query)
    context = await ds.get_context_from_vector_db(opt_user_query)
    human_message = HumanMessage(content="\n".join(context))
    human_message.content += f"\n\nUser's Query: {user_query}\n Answer:"

    CHUNK_SIZE: int = 128
    buffer: str = ""

    async for chunk in chat_response_llm.astream([system_message, human_message]):
        content = chunk.content
            
        # 2. Only accumulate if the chunk is a non-empty string
        if isinstance(content, str) and content:
            buffer += content
        else:
            continue
        # 3. Calculate and yield complete chunks
        # This calculates how many full CHUNK_SIZE batches we have
        chunks_to_yield = len(buffer) // CHUNK_SIZE
        
        if chunks_to_yield > 0:
            data_to_yield_len = chunks_to_yield * CHUNK_SIZE
            
            # Yield the complete part
            yield buffer[:data_to_yield_len]
            
            # Keep the leftover part in the buffer
            buffer = buffer[data_to_yield_len:]
            
        # 4. Yield the remaining content after the stream ends
    if buffer:
        yield buffer

    # # response = chat_response_llm.invoke([system_message, human_message])
    # CHUNK_SIZE = 128
    # buffer = ""
    # async for chunk in chat_response_llm.astream([system_message, human_message]):
    #     buffer += chunk.content
    #     while len(buffer) >= CHUNK_SIZE:
    #         dbg.info(f"Chat Response Chunk: {buffer[:CHUNK_SIZE]}")
    #         yield buffer[:CHUNK_SIZE]
    #         buffer = buffer[CHUNK_SIZE:]
    # if buffer:
    #     yield buffer

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