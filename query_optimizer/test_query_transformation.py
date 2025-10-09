import json
from datetime import datetime
from query_transformer import query_rewriter, query_expander, query_decomposer

# List of test queries
test_queries = [
    "List all companies in the automobile sector.",
    "Show holdings of Helios PMS in the month of August.",
    "What is the total market value of companies in the IT industry?",
    "Get the quantity and market value of HDFC Bank held by Axis PMS in July.",
    "Name of all finance sector companies owned by WhiteOak PMS.",
    "Compare the AUM percentage of HDFC Bank and ICICI Bank in August.",
    "List the companies from the automobile sector for the month of June.",
    "Which PMS holds the highest number of shares in the IT industry?",
    "Show the total quantity of shares across all PMS firms for Reliance Industries.",
    "Find all companies in the banking sector owned by Helios PMS and show their total market value in August."
]


def run_tests():
    print("\n================= Query Transformation Tests =================\n")

    results = []
    rewritten = ""
    expanded = ""
    decomposed = ""

    for i, query in enumerate(test_queries, start=1):
        print(f"\n================= Query {i} =================")
        print(f"User Query: {query}\n")

        rewritten = query_rewriter(query)
        # expanded = query_expander(query)
        # decomposed = query_decomposer(query)

        print(f"🔹 Rewritten Query:\n{rewritten}\n")
        print(f"🔹 Expanded Query:\n{expanded}\n")
        print(f"🔹 Decomposed Query:\n{decomposed}\n")
        print("------------------------------------------------------")

        results.append({
            "query_number": i,
            "user_query": query,
            "rewritten_query": rewritten,
            "expanded_query": expanded,
            "decomposed_query": decomposed
        })

    # Save to JSON
    json_filename = f"query_transformation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ All results saved to:\n- {json_filename}\n")


if __name__ == "__main__":
    run_tests()
