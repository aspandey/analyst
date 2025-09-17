from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import time
import os

import utils.app_config as cfg
import data_functions as df
dbg = cfg.DEBUG_APP

os.makedirs(cfg.DATA_DIR, exist_ok=True)
class QueryRequest(BaseModel):
    query: str

def initialize_documents_and_index():
    """
    Initializes and returns the documents and index required for query processing.

    Returns:
        tuple: (documents, index)
    """
    documents = df.data_create_documents_chunks(cfg.DATA_DIR, chunk_size=512, chunk_overlap=50)
    dbg.info(f"Loaded {len(documents)} documents from {cfg.DATA_DIR}")
    index = df.data_load_or_create_vector_database(documents)
    return documents, index

documents, index = initialize_documents_and_index()

# ==== FastAPI Server ====
app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "RAG server is up and running!"}

@app.post("/query-context")
async def get_query_context(req: QueryRequest) -> dict :
    """
    Retrieves relevant context for a given user query.
    Args:
        req (QueryRequest): Request containing the user's query.
    Returns:
        dict: Retrieved context.
    Raises:
        HTTPException: 400 if query is empty, 500 on retrieval error.
    """
    user_query = req.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        retrieved_context = df.data_get_query_context(user_query, documents, index, 20)
    except Exception as e:
        dbg.error(f"Error retrieving context: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve context.")
    return retrieved_context

    # Process:
    #     - Strips and validates the user's query.
    #     - Retrieves relevant context for the query.
    #     - Constructs a prompt using the context and user query.
    #     - Sends the prompt to an external generative API (Ollama) for answer generation.
    #     - Logs relevant information and handles errors appropriately.

@app.post("/query")
async def get_query_reply(req: QueryRequest):
    """
    Handles POST requests to generate an answer for a user's query using an external API.
    Args:
        req (QueryRequest): Request with the user's query.
    Returns:
        dict: Generated answer.
    Raises:
        HTTPException: On empty query or generation failure.
    """
    user_query = req.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Retrieve context for the query
    context_response = await get_query_context(req)
    context = context_response.get("retrieved_context", "")
    dbg.info(f"Retrieved context: {context}")

    response = await df.data_send_prompt_to_model(user_query, context)
    dbg.info(f"Response from Ollama API: {response.json()}")
    answer = response.json().get("response", "").strip()
    dbg.info(f"Generated answer: {answer}")
    
    if not answer:
        raise HTTPException(status_code=500, detail="No response generated.")

    return {"answer": answer}


# run this server using following command on terminal -
# uvicorn data_server:app --reload --port 5000