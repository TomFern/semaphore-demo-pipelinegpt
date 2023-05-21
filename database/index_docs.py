import openai
import pinecone
import pathlib
import tiktoken
import textwrap
import sys
import re
import os
from tqdm.auto import tqdm
from math import floor

# Pinecone settings
index_name = 'semaphore'
upsert_batch_size = 20

# OpenAI settings
max_tokens_model = 8191
embed_model = "text-embedding-ada-002"
encoding_model = "cl100k_base"

# https://platform.openai.com/docs/guides/embeddings/how-can-i-tell-how-many-tokens-a-string-has-before-i-embed-it
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def extract_yaml(text: str) -> str:
    """Returns list with all the YAML code blocks found in text."""
    matches = [m.group(1) for m in re.finditer("```yaml([\w\W]*?)```", text)]
    return matches


# Initialize connection to Pinecone 
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(api_key=api_key, enviroment=env)
index = pinecone.Index(index_name)

# read path to repo from CLI arguments
repo_path = sys.argv[1]
repo_path = os.path.abspath(repo_path)
repo = pathlib.Path(repo_path)

# Read YAMLs from Markdown data into memory
markdown_files = list(repo.glob("**/*.md")) + list(
    repo.glob("**/*.mdx")
)
print(f"Extracting YAML from Markdown files in {repo_path}")
new_data = []
for i in tqdm(range(0, len(markdown_files))):
    markdown_file = markdown_files[i]
    with open(markdown_file, "r") as f:
        relative_path = markdown_file.relative_to(repo_path)
        text = str(f.read())
        if text != '':
            yamls = extract_yaml(text)
            j = 0
            for y in yamls:
                j = j+1
                new_data.append({
                    "source": str(relative_path),
                    "text": y,
                    "id": "github.com/semaphore/docs/"+str(relative_path)+'['+str(j)+']'
                })

# Create embeddings and upsert the vectors to Pinecone
print(f"Creating embeddings and uploading vectors to database")
for i in tqdm(range(0, len(new_data), upsert_batch_size)):
    i_end = min(len(new_data), i+upsert_batch_size)
    meta_batch = new_data[i:i_end]
    ids_batch = [x['id'] for x in meta_batch]
    texts = [x['text'] for x in meta_batch]
    embedding = openai.Embedding.create(input=texts, engine=embed_model)
    embeds = [record['embedding'] for record in embedding['data']]
    # clean metadatab before upserting
    meta_batch = [{
        'id': x['id'],
        'text': x['text'],
        'source': x['source']
    } for x in meta_batch] 
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    index.upsert(vectors=to_upsert)

# Print final vector count
vector_count = index.describe_index_stats()['total_vector_count']
print(f"Database contains {vector_count} vectors.")
