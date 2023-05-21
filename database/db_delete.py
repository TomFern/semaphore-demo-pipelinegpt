import pinecone
import os

# Pinecone settings
index_name = "semaphore"

# Connect to Pinecone
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(api_key=api_key, environment=env)

pinecone.delete_index(index_name)