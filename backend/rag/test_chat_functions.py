from chat_functions import chat_with_user

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