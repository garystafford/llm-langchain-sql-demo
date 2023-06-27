# Natural Language Query Demo of Amazon RDS for PostgreSQL using OpenAI
# Author: Gary A. Stafford
# Date: 2023-06-25
# License: MIT
# OPENAI_API_KEY=<your_key>
# export SECRET_NAME="genai/rds/creds"
# export REGION_NAME="us-east-1"
# Usage: streamlit run app_nlq_rds_openai.py --server.runOnSave true

import json
import logging
import os

import boto3
import yaml
from botocore.exceptions import ClientError
from langchain import (FewShotPromptTemplate, PromptTemplate, SQLDatabase,
                       SQLDatabaseChain)
from langchain.chains.sql_database.prompt import (PROMPT_SUFFIX,
                                                  _postgres_prompt)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector.semantic_similarity import \
    SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from streamlit_chat import message

import streamlit as st

SECRET_NAME = os.environ.get("SECRET_NAME")
REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")


def main():
    st.set_page_config(page_title="Natural Language Query (NLQ) Demo")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # OpenAI API
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, verbose=True)

    # define datasource uri
    secret = get_secret(SECRET_NAME, REGION_NAME)
    rds_uri = get_datastore_credentials(secret)
    db = SQLDatabase.from_uri(rds_uri)

    # load examples for few-shot prompting
    examples = load_samples()

    chain = load_few_shot_chain(llm, db, examples)

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
        try:
            output = chain.run(query=user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
            logging.info(st.session_state["query"])
            logging.info(st.session_state["generated"])
        except Exception as exc:
            st.session_state.past.append(user_input)
            st.session_state.generated.append(
                "I do not have enough information to answer your question."
            )
            logging.error(exc)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


def get_secret(secret_name, region_name):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    secret = json.loads(get_secret_value_response["SecretString"])
    return secret


def get_datastore_credentials(secret):
    # SQLAlchemy 2.0 reference: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html
    # URI format: postgresql+psycopg2://user:pwd@hostname:port/dbname

    rds_username = secret["username"]
    rds_password = secret["password"]
    rds_endpoint = secret["host"]
    rds_port = secret["port"]
    rds_db_name = secret["dbname"]
    rds_db_name = "moma"
    rds_uri = f"postgresql+psycopg2://{rds_username}:{rds_password}@{rds_endpoint}:{rds_port}/{rds_db_name}"

    return rds_uri


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

    db_chain = SQLDatabaseChain.from_llm(
        llm,
        db,
        prompt=few_shot_prompt,
        use_query_checker=False,
        verbose=True,
        return_intermediate_steps=False,
    )

    return db_chain


def get_text(col1):
    with col1:
        input_text = st.text_input(
            "Ask a question:",
            "",
            key="query_text",
            placeholder="Your question here...",
            on_change=clear_text(),
        )


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
        [The Museum of Modern Art Collection](https://www.moma.org/collection/) contains two tables: 'artists' and 'artworks', with 121,211 pieces of artwork and 15,086 artists.
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
            # currently non-functional
            option = st.selectbox(
                "Choose a datasource:",
                (
                    "MoMA Collection (Amazon RDS for PostgreSQL)",
                    "TICKIT Sales (Amazon Redshift)",
                    "Classic Models (Amazon Aurora MySQL)",
                ),
                label_visibility=st.session_state.visibility,
                disabled=st.session_state.disabled,
            )
            with st.expander("Sample questions"):
                st.text(
                    """
                How many artists are there in the collection?
                How many pieces of artwork are there in the collection?
                How many artists are there whose nationality is French?
                How many artworks were created by Spanish artists?
                How many artist names start with the letter 'M'?
                What nationality of artists created the most number of artworks?
                How many artworks in the collection are by Claude Monet?
                What are the 3 oldest artworks in the collection?
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
