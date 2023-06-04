# query.py

# Pinecone database name, number of matched to retrieve
# cutoff similarity score, and how much tokens as context
index_name = 'semaphore'
context_cap_per_query = 30
match_min_score = 0.75
context_tokens_per_query = 3000

# OpenAI LLM model parameters
chat_engine_model = "gpt-3.5-turbo"
max_tokens_model = 4000
temperature = 0.2 
embed_model = "text-embedding-ada-002"
encoding_model_messages = "gpt-3.5-turbo-0301"
encoding_model_strings = "cl100k_base"

# Connect with Pinecone db and index
import pinecone
import os
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(api_key=api_key, environment=env)
index = pinecone.Index(index_name)

import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_model_strings)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages. Compatible with  model """

    try:
        encoding = tiktoken.encoding_for_model(encoding_model_messages)
    except KeyError:
        encoding = tiktoken.get_encoding(encoding_model_strings)

    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def get_prompt(query: str, context: str) -> str:
    """Return the prompt with query and context."""
    return (
        f"Create the continuous integration pipeline YAML code to fullfil the requested task.\n" +
        f"Below you will find some context that may help. Ignore it if it seems irrelevant.\n\n" +
        f"Context:\n{context}" +
        f"\n\nTask: {query}\n\nYAML Code:"
    )

def get_message(role: str, content: str) -> dict:
    """Generate a message for OpenAI API completion."""
    return {"role": role, "content": content}

import openai

def get_context(query: str, max_tokens: int) -> list:
    """Generate message for OpenAI model. Add context until hitting `context_token_limit` limit. Returns prompt string."""

    embeddings = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # search the database
    vectors = embeddings['data'][0]['embedding']
    embeddings = index.query(vectors, top_k=context_cap_per_query, include_metadata=True)
    matches = embeddings['matches']

    # filter and aggregate context
    usable_context = ""
    context_count = 0
    for i in range(0, len(matches)):

        source = matches[i]['metadata']['source']
        if matches[i]['score'] < match_min_score:
            # skip context with low similarity score
            continue
                    
        context = matches[i]['metadata']['text']
        token_count = num_tokens_from_string(usable_context + '\n---\n' + context)

        if token_count < context_tokens_per_query:
            usable_context = usable_context + '\n---\n' + context 
            context_count = context_count + 1

    print(f"Found {context_count} contexts for your query")

    return usable_context


def complete(messages):
    """Query the OpenAI model. Returns the first answer. """

    res = openai.ChatCompletion.create(
        model=chat_engine_model,
        messages=messages,
        temperature=temperature
    )
    return res.choices[0].message.content.strip()

import sys

query = sys.argv[1]

context = get_context(query, context_tokens_per_query)
prompt = get_prompt(query, context)

# initialize messages list to send to OpenAI API
messages = []
messages.append(get_message('user', prompt))
messages.append(get_message('system', 'You are a helpful assistant that writes YAML code for Semaphore continuous integration pipelines and explains them. Return YAML code inside code fences.'))

if num_tokens_from_messages(messages) >= max_tokens_model:
    raise Exception('Model token size limit reached') 

print("Working on your query... ")
answer = complete(messages)
print("Answer:\n")
print(answer)
messages.append(get_message('assistant', answer))

