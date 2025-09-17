import faiss
import numpy as np
import requests
import os
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import HTTPException

import utils.app_config as cfg
dbg = cfg.DEBUG_APP


async def data_send_prompt_to_model(user_query: str, context: str) -> requests.Response:
    """
    Sends a prompt constructed from the user's query and provided context to a generative model API and returns the response.

    Args:
        user_query (str): The user's question to be answered by the model.
        context (str): The context information to be used by the model for generating an answer.

    Returns:
        requests.Response: The HTTP response object returned by the generative model API.

    Raises:
        HTTPException: If the API request fails or an error occurs during the generation process.
    """
    # Prepare prompt for generation
    prompt = (
        "Use the following context to answer the question.\n"
        f"Context:\n{context}\n"
        f"Question: {user_query}\n"
        "Answer:"
    )
    dbg.info(f"Generated prompt: {prompt}")
    try:
        response = requests.post(
            f"{cfg.OLLAMA_API}/generate",
            json={
                "model": cfg.GENERATIVE_MODEL,
                "prompt": prompt,
                "temperature": 0.5,
                "num_predict": 500,
                "stream": False,
                "top_k": 100,
                "top_p": 0.9,
            }
        )
        response.raise_for_status()
    except Exception as e:
        dbg.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate an answer")

    return response

def data_create_documents_chunks(folder_path: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list:
    """
    Reads PDF files from the specified folder and splits their content into chunks using LangChain APIs.

    Args:
        folder_path (str): Path to the folder containing PDF files.
        chunk_size (int): The size of each chunk. Default is 512 characters.
        chunk_overlap (int): The number of overlapping characters between chunks. Default is 50.

    Returns:
        list: A list of text chunks from all PDFs.
    """
    if not os.path.isdir(folder_path):
        # dbg.error(f"Folder path does not exist: {folder_path}")
        raise ValueError(f"Folder path does not exist: {folder_path}")

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            dbg.info(f"Loaded {len(documents)} documents from {filename}")
            for doc in documents:
                # dbg.info(f"Processing file: {filename}, page content length: {len(doc.page_content)}")
                chunks = splitter.split_text(doc.page_content)
                all_chunks.extend(chunks)
    return all_chunks



def data_create_embedding_vectors(text: str) -> np.ndarray:
    """
    Generates an embedding vector for the given text using the Ollama API and EMBEDDING_MODEL.

    Args:
        text (str): The input text to embed.

    Returns:
        np.ndarray: The embedding vector as a NumPy array.

    Raises:
        ValueError: If the API response does not contain an 'embedding' key.
        requests.RequestException: If the API request fails.
    """
    try:
        response = requests.post(
            f"{cfg.OLLAMA_API}/embeddings",
            json={
                "model": cfg.EMBEDDING_MODEL,
                "prompt": text
            },
            timeout=100
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get('embedding')
        if embedding is None:
            dbg.error("No 'embedding' key in Ollama API response")
            raise ValueError("No 'embedding' key in response")
        return np.array(embedding, dtype=np.float32)
    except requests.RequestException as e:
        dbg.error(f"Failed to get embedding from Ollama API: {e}")
        raise

# Create FAISS vector database of indicies
# Load or create the FAISS index:
# - If the index file exists, load it from disk.
# - Otherwise, compute embeddings for all documents, create a FAISS index with Euclidean distance,
#   add the embeddings, and save the index to disk.

def data_load_or_create_vector_database(documents: list) -> faiss.Index:
    """
    Loads a FAISS index from disk if it exists, otherwise creates a new index from the provided documents.

    Args:
        documents (list): List of document strings to embed and index.

    Returns:
        faiss.Index: The loaded or newly created FAISS index.
    """
    index_file = cfg.DATA_INDEX_FILE

    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        dbg.info(f"Loaded FAISS index from {index_file} ({type(index_file)})")
    else:
        if not documents:
            dbg.error("Document list is empty. Cannot create FAISS index.")
            raise ValueError("Document list cannot be empty when creating a new FAISS index.")
        embeddings = np.array([data_create_embedding_vectors(doc) for doc in documents])
        dbg.info(f"Generated embeddings for {len(documents)} documents")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        index.add(embeddings)
        faiss.write_index(index, index_file)
        dbg.info(f"Created and saved new FAISS index at {index_file}")

    return index


def data_get_query_context(query: str, documents: list, index: faiss.Index, k: int = 20) -> dict:
    """
    Retrieve relevant context for a given query using a FAISS vector index.

    Args:
        query (str): The input query string.
        documents (list): List of document strings.
        index (faiss.Index): FAISS index containing document embeddings.
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        dict: Dictionary with the retrieved context.
    """
    if not query:
        dbg.error("Query cannot be empty")
        raise ValueError("Query cannot be empty")
    if not documents:
        dbg.error("Documents list cannot be empty or None")
        raise ValueError("Documents list cannot be empty or None")
    if index is None:
        dbg.error("FAISS index is not provided")
        raise ValueError("FAISS index is required")

    query_embed = data_create_embedding_vectors(query).reshape(1, -1)
    _, neighbor_indices = index.search(query_embed, k=min(k, len(documents)))
    retrieved_context = "\n".join([documents[i] for i in neighbor_indices[0] if i < len(documents)])

    return {"retrieved_context": retrieved_context}