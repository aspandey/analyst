EMBEDDING_MODEL="nomic-embed-text"
# GENERATIVE_MODEL = "phi3:mini"
# GENERATIVE_MODEL = "gemma3:1b"
GENERATIVE_MODEL="llama3.2:latest"
# ollama run $EMBEDDING_MODEL
ollama run $GENERATIVE_MODEL
