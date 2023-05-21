# semaphore-demo-pipelinegpt

Demo project to build an enriched ChatGPT responses using embeddings.

## Requirements

- Python 3
- A OpenAI API account (paid)
- A Starter Plan Pinecone account (free)

## Installation

1. Clone the repository
2. Create a virtualenv for Python:
    ```bash
    $ virtualenv venv
    $ source venv/bin/activate
    ```
3. Install dependencies
    ```bash
    $ pip install -r requirements.txt
    ```
4. Initialize environment file. Add your Pinecone and OpenAI API Keys
    ```bash
    $ cp env-example .env
    $ nano .env
    ```

## Create embeddings database

1. Clone the Semaphore docs repository
    ```bash
    $ git clone https://github.com/semaphoreci/docs.git
    ```
2. Create a Pinecone database
    ```bash
    $ source venv/bin/activate
    $ source .env
    $ cd database
    $ python db_create.py
    ```
3. Create embeddings and upload them to Pinecone. 
    ```bash
    $ source venv/bin/activate
    $ source .env
    $ cd database
    $ python index_docs.py /path/repository/semaphoreci/docs

    Extracting YAML from Markdown files in /Users/tom/r/docs
    100%|████████████████████████| 164/164 [00:00<00:00, 8206.95it/s]
    Creating embeddings and uploading vectors to database
    100%|████████████████████████| 4/4 [00:09<00:00,  2.27s/it]
    Database contains 79 vectors.
    ```

## Run your query

```bash
$ source venv/bin/activate
$ source .env
$ python query.py "Create a continuous integration pipeline to build and upload a Docker image to Docker Hub"
EXAMPLE ANSWER
```

