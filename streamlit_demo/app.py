# Natural Language Query Demo of Amazon Redshift using Streamlit and OpenAI
# Author: Gary A. Stafford
# Date: 2023-06-09
# License: MIT

# sudo yum update -y && sudo yum install gcc gcc-c++ make
# python3 -m pip install SQLAlchemy==1.4.48 -q # older version required for sqlalchemy-redshift
# python3 -m pip install langchain openai sqlalchemy-redshift -Uq
# python3 -m pip install streamlit streamlit-chat psycopg2-binary chromadb python-dotenv -Uq
# python3 -m pip install pyyaml -q
# python3 -m pip install sentence-transformers -Uq --no-cache-dir #--force-reinstall
# pip list | grep "langchain\|openai\|sentence-transformers\|SQLAlchemy"

# streamlit run app.py --server.runOnSave true

import os

import yaml
from dotenv import load_dotenv
from langchain import (
    FewShotPromptTemplate,
    PromptTemplate,
    SQLDatabase,
    SQLDatabaseChain,
)
from langchain.chains import SQLDatabaseSequentialChain
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _postgres_prompt
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.example_selector.semantic_similarity import (
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma
from streamlit_chat import message

import streamlit as st


def main():
    load_dotenv()

    # llm = OpenAI(model_name="text-davinci-003", temperature=0, verbose=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)
    redshift_endpoint = get_datastore_credentials()
    db = SQLDatabase.from_uri(redshift_endpoint)
    examples = load_samples()

    st.set_page_config(
        page_title="LangChain Natural Language Query Demo", page_icon=":robot:"
    )

    # chain = load_chain(llm, db)
    chain = load_few_shot_chain(llm, db, examples)

    st.header("LangChain NLQ Demo")
    st.subheader("Datasource")
    st.text("Amazon Redshift TICKIT database")
    st.subheader("Sample Questions")
    st.text(
        """
    • How many venues are there in state of NY?
    • How many customers made a purchase in May 2022?
    • What were the total sales in September 2022?
    • What are the largest 3 events based on all time gross sales?
    • Who are the top 5 sellers based on all time gross sales?
    • Which venue hosted the most events?
    """
    )

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    user_input = get_text()

    if user_input:
        output = chain.run(query=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


def get_datastore_credentials():
    redshift_host = os.environ.get("REDSHIFT_HOST")
    redshift_port = os.environ.get("REDSHIFT_PORT")
    redshift_database = os.environ.get("REDSHIFT_DATABASE")
    redshift_username = os.environ.get("REDSHIFT_USERNAME")
    redshift_password = os.environ.get("REDSHIFT_PASSWORD")
    redshift_endpoint = f"redshift+psycopg2://{redshift_username}:{redshift_password}@{redshift_host}:{redshift_port}/{redshift_database}"

    return redshift_endpoint


def load_samples():
    # Use the corrected examples for few shot prompt examples
    sql_samples = None

    with open("../few_shot_examples/sql_examples_redshift_slim.yaml", "r") as stream:
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


def load_chain(llm, db):
    db_chain = SQLDatabaseSequentialChain.from_llm(
        llm, db, verbose=True, use_query_checker=False
    )
    return db_chain


def get_text():
    input_text = st.text_input(
        "Question:", "", key="query", placeholder="Your question..."
    )
    return input_text


if __name__ == "__main__":
    main()
