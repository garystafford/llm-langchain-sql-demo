# Performing NLQ on Amazon RDS using Amazon SageMaker, LangChain, and LLMs

Source code for the upcoming blog post, [Perform Natural Language Queries of Amazon RDS using Amazon SageMaker, LangChain, and LLMs](#). Learn to use LangChain's SQL Database Chain and Agent with various LLMs to perform Natural Language Queries (NLQ) of an Amazon RDS for PostgreSQL database.

## dotenv

Your `.env` files should look as follows:

```ini
# API Keys
OPENAI_API_KEY=

# RDS Connection 
RDS_ENDPOINT=
RDS_PORT=
RDS_USERNAME=
RDS_PASSWORD=
RDS_DB_NAME=
```

## Checking RDS Connection from SageMaker Notebook

```sh
dig +short txt ch whoami.cloudflare @1.0.0.1

curl -v ******.******.us-east-1.rds.amazonaws.com:5432
```

## Notebook Formatting

Using `jupyter-black` for formatting notebooks.

```sh
pip install "black[jupyter]"

black *.ipynb
```