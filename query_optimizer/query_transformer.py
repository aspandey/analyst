from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
import asyncio

from query_optimizer.query_prompt import QUERY_TRANS_PROMPT, QUERY_CLASSIFIER_PROPMT

llm = ChatOllama(
    model="llama3.2:latest",
    # model = "deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.1,
    reasoning=False
    )

async def query_classifier(user_query: str) -> str:
    """
    Classify the user query into one of the transformation types:
    'rewrite', 'expand', or 'decompose', using an LLM.
    """
    system_message = SystemMessage(content=QUERY_CLASSIFIER_PROPMT)
    human_message = HumanMessage(content=user_query)

    response = llm.invoke([system_message, human_message])
    res = response.content
    classification  = res

    if classification not in ["rewrite", "expand", "decompose"]:
        classification = "rewrite"

    return classification


async def query_transformer(user_input: str, transformer: str) -> str:
    system_message = SystemMessage(content=transformer)
    human_message = HumanMessage(content=user_input)

    response = llm.invoke([system_message, human_message])
    res = response.content
    if not isinstance(res, str):
        res = str(res)
    return res


async def query_optimizer(user_query: str) -> str:

    optimized_query: str = ""

    query_class = query_classifier(user_query)
    print(f"🔹 Query class:\n{query_class}\n")

    if query_class not in ["rewrite", "expand", "decompose"]:
        query_class = "rewrite"

    optimized_query = await query_transformer(user_query, QUERY_TRANS_PROMPT[query_class])
 
    return optimized_query


# For Manual input testing.
# p3 query_optimizer/query_transformer.py

async def main ():
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        query_class = await query_classifier(user_query)
        print(f"Classification: {query_class}")
        if query_class not in ["rewrite", "expand", "decompose"]:
            query_class = "rewrite"
        else:
            print("Unknown classification. No transformation applied.")
        
        optimized_query = query_transformer(user_query, QUERY_TRANS_PROMPT[query_class])
        
if __name__ == "__main__":
    asyncio.run(main())