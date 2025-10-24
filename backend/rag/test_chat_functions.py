from rag.chat_functions import app_stocks_info, stream_response_with_context
import asyncio
from prompts.chat_prompt import FINANCE_EXPERT_SYSTEM_PROMPTS
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage


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

async def test_model_call ():

    test_query = "list of companies working in banking sector and held by helios pms. How many of hdfc bank shares are held by helios pms"
    with open('/Users/soumyashreeswain/factory/public/analyst/test-data/example-logs.txt', 'r') as file:
        logs = file.readlines()
    logs = [log.strip() for log in logs]
    new_logs = logs[::-1]  # Reverse the logs to have the latest first
    print("Logs read from file:", new_logs)
    full_response = ""
    async for chunk in stream_response_with_context(new_logs, test_query, SystemMessage(content=FINANCE_EXPERT_SYSTEM_PROMPTS["V2"])):
            # yield chunk
            full_response += chunk
    
    print(f"\nTotal collected response length: {full_response} characters")


    

if __name__ == "__main__":
    asyncio.run(test_model_call())