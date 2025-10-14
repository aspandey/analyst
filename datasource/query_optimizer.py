from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from query_optimizer.query_transformer import query_rewriter, query_expander, query_decomposer, query_classifier

import json
import asyncio
    
llm = ChatOllama(
    model="llama3.2:latest",
    # model = "deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.2,
    reasoning=False
    )

async def query_optimizer(user_query: str) -> str:

    optimized_query: str = ""

    query_class = query_classifier(user_query)
    print(f"🔹 Query class:\n{query_class}\n")
    if query_class == "rewrite":
        optimized_query = query_rewriter(user_query)
    elif query_class == "expand":
        optimized_query = query_expander(user_query)
    elif query_class == "decompose":
        optimized_query = query_decomposer(user_query)
    
    return optimized_query

async def main():
    pass

if __name__ == "__main__":
    asyncio.run(main())

