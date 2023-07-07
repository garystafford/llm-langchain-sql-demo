# Natural Language Query Demo of Amazon RDS for PostgreSQL using SageMaker FM Endpoint
# Author: Gary A. Stafford
# Date: 2023-06-25
# License: MIT
# Application expects the following environment variables:
# export ENDPOINT_NAME="hf-text2text-flan-t5-xxl-fp16"
# export REGION_NAME="us-east-1"
# export SECRET_NAME="genai/rds/creds"
# Usage: streamlit run app.py --server.runOnSave true

import json
import logging
import os

import boto3
import streamlit as st
import yaml
from botocore.exceptions import ClientError
from langchain import (
    FewShotPromptTemplate,
    PromptTemplate,
    SQLDatabase,
    SQLDatabaseChain,
)
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _postgres_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.prompts.example_selector.semantic_similarity import (
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma
from streamlit_chat import message

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME")
REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")
SECRET_NAME = os.environ.get("SECRET_NAME")


def main():
    st.set_page_config(page_title="Natural Language Query (NLQ) Demo")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Amazon SageMaker JumpStart Endpoint
    content_handler = ContentHandler()

    parameters = {
        "max_length": 2048,
        "temperature": 0.0,
    }

    llm = SagemakerEndpoint(
        endpoint_name=ENDPOINT_NAME,
        region_name=REGION_NAME,
        model_kwargs=parameters,
        content_handler=content_handler,
    )

    # define datasource uri
    secret = get_secret(SECRET_NAME, REGION_NAME)
    rds_uri = get_rds_uri(secret)
    db = SQLDatabase.from_uri(rds_uri)

    # load examples for few-shot prompting
    examples = load_samples()

    sql_db_chain = load_few_shot_chain(llm, db, examples)

    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "query" not in st.session_state:
        st.session_state["query"] = []

    if "query_text" not in st.session_state:
        st.session_state["query_text"] = []

    # define streamlit colums
    col1, col2 = st.columns([4, 1], gap="large")

    # build the streamlit sidebar
    build_sidebar()

    # build the main app ui
    build_form(col1, col2)

    # get the users query
    get_text(col1)
    user_input = st.session_state["query"]

    if user_input:
        st.session_state.past.append(user_input)
        try:
            output = sql_db_chain.run(query=user_input)
            st.session_state.generated.append(output)
            logging.info(st.session_state["query"])
            logging.info(st.session_state["generated"])
        except Exception as exc:
            st.session_state.generated.append(
                "I'm sorry, I was not able to answer your question."
            )
            logging.error(exc)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i],
                    key=str(i),
                    is_user=False,
                    avatar_style="icons",
                    seed="459"
                   )
            message(
                st.session_state["past"][i],
                is_user=True, 
                key=str(i) + "_user",
                avatar_style="icons",
                seed="158"
            )


def get_secret(secret_name, region_name):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    return json.loads(get_secret_value_response["SecretString"])


def get_rds_uri(secret):
    # SQLAlchemy 2.0 reference: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html
    # URI format: postgresql+psycopg2://user:pwd@hostname:port/dbname

    rds_username = secret["username"]
    rds_password = secret["password"]
    rds_endpoint = secret["host"]
    rds_port = secret["port"]
    rds_db_name = secret["dbname"]
    rds_db_name = "moma"
    return f"postgresql+psycopg2://{rds_username}:{rds_password}@{rds_endpoint}:{rds_port}/{rds_db_name}"


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]


def load_samples():
    # Use the corrected examples for few-shot prompting examples
    sql_samples = None

    with open("sql_examples_postgresql.yaml", "r") as stream:
        sql_samples = yaml.safe_load(stream)

    return sql_samples


def load_few_shot_chain(llm, db, examples):
    example_prompt = PromptTemplate(
        input_variables=["table_info", "input", "sql_cmd", "sql_result", "answer"],
        template="{table_info}\n\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {sql_result}\nAnswer: {answer}",
    )

    local_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        local_embeddings,
        Chroma,
        k=min(3, len(examples)),
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_postgres_prompt + "Here are some examples:",
        suffix=PROMPT_SUFFIX,
        input_variables=["table_info", "input", "top_k"],
    )

    return SQLDatabaseChain.from_llm(
        llm,
        db,
        prompt=few_shot_prompt,
        use_query_checker=True,
        verbose=True,
        return_intermediate_steps=False,
    )


def get_text(col1):
    with col1:
        input_text = st.text_input(
            "Ask a question:",
            "",
            key="query_text",
            placeholder="Your question here...",
            on_change=clear_text(),
        )
        logging.info(input_text)


def clear_text():
    st.session_state["query"] = st.session_state["query_text"]
    st.session_state["query_text"] = ""


def build_sidebar():
    with st.sidebar:
        st.title("Technologies")

        st.subheader("Natural Language Query (NLQ)")
        st.write(
            """
        [Natural language query (NLQ)](https://www.yellowfinbi.com/glossary/natural-language-query) enables analytics users to ask questions of their data. It parses for keywords and generates relevant answers sourced from related databases, with results typically delivered as a report, chart or textual explanation that attempt to answer the query, and provide depth of understanding.
        """
        )

        st.subheader("MoMa Collection Database")
        st.write(
            """
        [The Museum of Modern Art Collection](https://www.moma.org/collection/) contains two tables: 'artists' and 'artworks', with ~121,000 pieces of artwork and ~15,000 artists.
        """
        )

        st.subheader("Amazon SageMaker Studio")
        st.write(
            """
        [Amazon SageMaker Studio](https://aws.amazon.com/sagemaker/studio/) is a fully integrated development environment (IDE) where you can perform all machine learning (ML) development steps, from preparing data to building, training, and deploying your ML models, including [JumpStart Foundation Models](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models.html).
        """
        )

        st.subheader("LangChain")
        st.write(
            """
        [LangChain](https://python.langchain.com/en/latest/index.html) is a framework for developing applications powered by language models.
        """
        )

        st.subheader("Chroma")
        st.write(
            """
        [Chroma](https://www.trychroma.com/) is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.
        """
        )

        st.subheader("Streamlit")
        st.write(
            """
        [Streamlit](https://streamlit.io/) is an open-source app framework for Machine Learning and Data Science teams. Streamlit turns data scripts into shareable web apps in minutes. All in pure Python. No front-end experience required.
        """
        )


def build_form(col1, col2):
    with col1:
        with st.container():
            st.title("Natural Language Query (NLQ) Demo")
            st.subheader("Ask questions of your data using natural language.")

        with st.container():
            with st.expander("Sample questions (copy and paste)"):
                st.text(
                    """
                How many artists are there in the collection?
                How many pieces of artwork are there in the collection?
                How many paintings are in the collection?
                How many artists are there whose nationality is French?
                How many artworks were created by Spanish artists?
                How many artist names start with the letter 'M'?
                What nationality of artists created the most number of artworks?
                How many artworks are by the artist, 'Claude Monet'?
                What are the 3 oldest artworks in the collection? Return the title and date.
                """
                )
    with col2:
        with st.container():
            st.button("clear chat", on_click=clear_session)


def clear_session():
    for key in st.session_state.keys():
        del st.session_state[key]


if __name__ == "__main__":
    main()
