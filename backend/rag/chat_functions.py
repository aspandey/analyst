from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

import weaviate_database.db_collection as ds
import query_optimizer.query_transformer as qo
from prompts.chat_prompt import FINANCE_EXPERT_SYSTEM_PROMPTS
from debug.logger_config import dbg
from typing import AsyncGenerator
from langchain_core.messages import SystemMessage, HumanMessage


chat_response_llm = ChatOllama(
    model="llama3.2:latest",
    # model = "deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.4,
    reasoning=False
    )

# Add this helper at module level so you can test it directly
async def stream_response_with_context(
    context: list[str], user_query: str, system_message: SystemMessage
) -> AsyncGenerator[str, None]:
    human_message = HumanMessage(
        content="\n".join(context) + f"\n\nUser's Query: {user_query}\nAnswer:"
    )

    async for chunk in chat_response_llm.astream([system_message, human_message]):
        content = getattr(chunk, "content", None)
        if not isinstance(content, str) or not content:
            continue
        yield content


async def app_stocks_info(user_query: str) -> AsyncGenerator[str, None]:
    """
    Asynchronously streams an AI-generated response to a user query 
    using a Retrieval-Augmented Generation (RAG) workflow.

    Args:
        user_query (str): The user's input question or message.

    Yields:
        str: Incremental chunks of the generated response text.

    Workflow:
        1. Optimizes the user's query for better retrieval relevance.
        2. Fetches related context from the vector database.
        3. Streams an LLM-generated answer using the retrieved context.
    """
    system_message = SystemMessage(content=FINANCE_EXPERT_SYSTEM_PROMPTS["V2"])

    # 1. Optimize user query and fetch contextual information
    optimized_query = await qo.query_optimizer(user_query)
    context = await ds.get_context_from_vector_db(optimized_query)
    new_context = context[::-1]
    async for chunk in stream_response_with_context(new_context, user_query, system_message):
        yield chunk


############# Test code for chat_with_user ############# 
# p3 -m rag.app

async def main():
    while True:
        print("================================== New Query =================================")
        user_input = input("Enter your query (type 'exit' to quit): ")

        if user_input.lower() == "exit":
            print("Exiting chat.")
            break
        async def stream_response():
            async for chunk in app_stocks_info(user_input):
                yield chunk

        printed_chunk = stream_response()
        print(f"\n +++++++++++++++++++++ AI Message +++++++++++++++++++++ \n {printed_chunk} \n ")

import asyncio

if __name__ == "__main__":
    asyncio.run(main())    