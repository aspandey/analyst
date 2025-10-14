import json
from datetime import datetime
from query_transformer import query_rewriter, query_expander, query_decomposer, query_classifier

# List of test queries
test_queries = [
    # "List all companies in the automobile sector.",
    # "Show holdings of Helios PMS in the month of August.",
    # "What is the total market value of companies in the IT industry?",
    # "Get the quantity and market value of HDFC Bank held by Axis PMS in July.",
    # "Name of all finance sector companies owned by WhiteOak PMS.",
    # "Compare the AUM percentage of HDFC Bank and ICICI Bank in August.",
    # "List the companies from the automobile sector for the month of June.",
    # "Which PMS holds the highest number of shares in the IT industry?",
    # "Show the total quantity of shares across all PMS firms for Reliance Industries.",
    "Find all companies in the banking sector owned by Helios PMS and show their total market value in August.",
    "I want to see all the companies that might be related to finance or banking, and I’m particularly interested in those managed by Helios PMS in recent months.",
    "Can you tell me which companies were held in July or August by Helios or Axis PMS?",
    "List the companies that are in either IT or tech-related sectors, and also mention the month of their holdings.",
    "Show me companies with significant holdings in the last few months and their approximate market values.",
    "I am curious about companies in finance or banking sectors, especially the ones with high AUM percentage, and managed by any of the major PMS firms.",
    "Which companies that HDFC Bank or ICICI Bank have invested in, and what was the quantity of shares in July?",
    "Give me the list of companies from multiple sectors like finance, IT, and healthcare, and include information about the PMS managing them.",
    "I just want to know the companies managed by Helios PMS last month, but I don’t care about their exact quantity.",
    "Compare all companies in the banking sector between Helios PMS and Axis PMS, and provide data for the recent quarter.",
    "I’m trying to figure out which companies in the finance industry were part of Helios PMS portfolio, and also check for the months of July and August, including their share quantities and market values if possible.",
]

def run_tests():
    print("\n================= Query Transformation Tests =================\n")

    results = []

    for i, query in enumerate(test_queries, start=1):
        # query_class = ""
        rewritten = ""
        expanded = ""
        decomposed = ""
        print(f"\n================= Query {i} =================")
        print(f"User Query: {query}\n")

        query_class = query_classifier(query)
        print(f"🔹 Query class:\n{query_class}\n")
        if query_class == "rewrite":
            rewritten = query_rewriter(query)
        elif query_class == "expand":
            expanded = query_expander(query)
        elif query_class == "decompose":
            decomposed = query_decomposer(query)

        print("------------------------------------------------------")

        results.append({
            "query_number": i,
            "user_query": query,
            "rewritten_query": rewritten,
            "expanded_query": expanded,
            "decomposed_query": decomposed,
            "query_class" : query_class
        })

    json_filename = f"query_transformation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ All results saved to:\n- {json_filename}\n")


if __name__ == "__main__":
    run_tests()
