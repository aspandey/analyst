from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
    
llm = ChatOllama(
    model="llama3.2:latest",
    # model = "deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.1,
    reasoning=False
    )

def query_classifier(user_query: str) -> str:
    """
    Classify the user query into one of the transformation types:
    'rewrite', 'expand', or 'decompose', using an LLM.
    """
    system_message = SystemMessage(content=
        "You are a query classifier for a vector search engine. "
        "Your task is to categorize the user's query into one of the following transformations:\n"
        "1. rewrite - Optimize verbose, conversational, or unclear queries into short, concise, schema-aligned keywords.\n"
        "2. expand - Enrich short or vague queries with related keywords, synonyms, or semantically relevant phrases.\n"
        "3. decompose - Split multi-intent or complex queries into smaller, focused sub-queries.\n\n"
        "Output only one word: rewrite, expand, or decompose."
    )

    human_message = HumanMessage(content=user_query)

    response = llm.invoke([system_message, human_message])
    res = response.content
    classification  = res
    # Ensure it returns a valid classification
    if classification not in ["rewrite", "expand", "decompose"]:
        # Default fallback
        classification = "rewrite"

    return classification


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
        "The vector database has the following schema: "
        "`company_or_stock_name`, `industry_sector`, `data_month`, `portfolio_management_services_name`. "
        "Expand the user's query by adding relevant keywords, synonyms, and related phrases "
        "to improve semantic recall without changing its intent. "
        "that directly map to these fields and can be used for keyword search. "
        "Output should be expended queries only. No conversation. "
        "Keep the output short and focused — a comma-separated list of meaningful terms only."
        "Provide maximum 3 comma-separated items in the list"
        "Example: 'companies in finance sector' → 'finance companies, financial institutions, banking sector, finance industry'."
    )

    human_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message, human_message])
    return response.content

def query_decomposer(user_input: str) -> str:
    system_message = SystemMessage(content=
        "You are a Query Decomposer for a vector search engine. Break complex user queries into smaller, focused sub-queries. "
        "The vector database has the following schema: "
        "`company_or_stock_name`, `industry_sector`, `data_month`, `portfolio_management_services_name`. "
        "Each sub-query should represent a single intent or a distinct piece of information that can be searched independently. "
        "Keep each sub-query short, clear, and in logical order. "
        "Avoid conversational words, filler text, or long sentences. "
        "Output should be sub queries only. No conversation. Maximum 3 subqueries."
        "Example: 'Show me all companies in the finance sector and their total quantity in July' → "
        "1. companies in finance sector. "
        "2. total quantity of companies in July."
    )

    human_message = HumanMessage(content=user_input)
    response = llm.invoke([system_message, human_message])
    return response.content


# For Manual input testing.
# p3 query_optimizer/query_transformer.py

if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        classification = query_classifier(user_query)
        print(f"Classification: {classification}")

        if classification == "rewrite":
            result = query_rewriter(user_query)
            print("Rewritten Query:", result)
        elif classification == "expand":
            result = query_expander(user_query)
            print("Expanded Query:", result)
        elif classification == "decompose":
            result = query_decomposer(user_query)
            print("Decomposed Sub-Queries:", result)
        else:
            print("Unknown classification. No transformation applied.")