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
$ python query.py "Create a CI pipeline to build and upload a Docker image to Docker Hub"
Found 22 contexts for your query
Working on your query...

Answer:

version: v1.0
name: Docker Build and Push
agent:
  machine:
    type: e1-standard-2
    os_image: ubuntu1804

blocks:
  - name: "Build and Push Docker Image"
    task:
      jobs:
        - name: "Build and Push"
          commands:
            - checkout
            - docker build -t <repository>/<image>:<tag> .
            - echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
            - docker push <repository>/<image>:<tag>
      secrets:
        - name: dockerhub
promotions:
  - name: Deploy to Kubernetes
    pipeline_file: deploy-k8s.yml
    auto_promote:
      when: "result = 'passed'"
```

