import openai
from tqdm.auto import tqdm
import pathlib
from math import floor
import tiktoken
import textwrap
# from time import sleep
import sys
import re
import pinecone
import os

# Pinecone settings
index_name = 'semaphore'
upsert_batch_size = 20

# OpenAI settings
max_tokens_model = 8191
# max_tokens_model = 8000
embed_model = "text-embedding-ada-002"
encoding_model = "cl100k_base"

# https://platform.openai.com/docs/guides/embeddings/how-can-i-tell-how-many-tokens-a-string-has-before-i-embed-it
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def partition_text(text):
    partitioned = []
    tokens = num_tokens_from_string(text)
    if tokens <= max_tokens_model:
        return [text]
    else:
        chunk_size = floor(max_tokens_model * len(text) / tokens)
        return textwrap.wrap(text, chunk_size)

def extract_yaml(text: str) -> str:
    """Returns list with all the YAML code blocks found in text."""
    matches = [m.group(1) for m in re.finditer("```yaml([\w\W]*?)```", text)]
    return matches

# def split_markdown_sections(markdown_text):
#     # Regular expression pattern to match sections starting with '#'
#     pattern = r'^#\s+(.*)$'

#     # Split the markdown text into sections using the pattern
#     sections = re.split(pattern, markdown_text, flags=re.MULTILINE)

#     # Remove any leading or trailing empty sections
#     sections = [section.strip() for section in sections if section.strip()]

#     return sections

# Initialize connection to Pinecone 
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(api_key=api_key, enviroment=env)
index = pinecone.Index(index_name)

# read path to repo from CLI arguments
repo_path = sys.argv[1]
repo_path = os.path.abspath(repo_path)
# repo_path = "/Users/tom/r/semaphore-docs"
repo = pathlib.Path(repo_path)

markdown_files = list(repo.glob("**/*.md")) + list(
    repo.glob("**/*.mdx")
)

# print(repo_path)
# print(repo)
# print(markdown_files)
# exit(0)

# Read YAMLs from Markdown data into memory
print(f"Extracting YAML from Markdown files in {repo_path}")
new_data = []
for i in tqdm(range(0, len(markdown_files))):
    markdown_file = markdown_files[i]
    with open(markdown_file, "r") as f:
        relative_path = markdown_file.relative_to(repo_path)
        text = str(f.read())
        if text != '':
            # print("Processing file: " + str(relative_path))
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
    # find end of batch
    i_end = min(len(new_data), i+upsert_batch_size)
    meta_batch = new_data[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # empty files create a problem
    # create embeddings (try-except added to avoid RateLimitError)
    # print(texts)
    # print(ids_batch)
    embedding = openai.Embedding.create(input=texts, engine=embed_model)

    # try:
    #     embedding = openai.Embedding.create(input=texts, engine=embed_model)
    # except:
    #     done = False
    #     while not done:
    #         sleep(5)
    #         try:
    #             embedding = openai.Embedding.create(input=texts, engine=embed_model)
    #             done = True
    #         except:
    #             pass
    embeds = [record['embedding'] for record in embedding['data']]
    meta_batch = [{
        'id': x['id'],
        'text': x['text'],
        'source': x['source']
    } for x in meta_batch] 
    # cleanup metadata
    # meta_batch = []
    # meta_batch = [{
        # 'text': x['text'],
        # 'source': x['source'] #,
        # 'title': x['title'],
        # 'text': x['text'],
        # 'url': x['url'],
        # 'published': x['published'],
        # 'channel_id': x['channel_id']
    # } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)

vector_count = index.describe_index_stats()['total_vector_count']
print(f"Database contains {vector_count} vectors.")
