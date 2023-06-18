# Natural Language Query (NLQ) Demo of Amazon Redshift using OpenAI's LLMs, LangChain, and Streamlit.
# Author: Gary A. Stafford
# Date: 2023-06-14
# License: MIT

import os

import yaml
from dotenv import load_dotenv
from langchain import (
    FewShotPromptTemplate,
    PromptTemplate,
    SQLDatabase,
    SQLDatabaseChain,
)
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _postgres_prompt
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector.semantic_similarity import (
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma
from streamlit_chat import message

import streamlit as st


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load environment variables
    load_dotenv()

    # select llm to use
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)

    # define datasource uri
    redshift_endpoint = get_datastore_credentials()
    db = SQLDatabase.from_uri(redshift_endpoint)

    # load examples for few shot prmopting
    examples = load_samples()

    st.set_page_config(page_title="Natural Language Query (NLQ) Demo")

    chain = load_few_shot_chain(llm, db, examples)

    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    # hide streamlit default features
    # hide_menu_style = """
    #     <style>
    #     #MainMenu {visibility: hidden; }
    #     footer {visibility: hidden;}
    #     </style>
    #     """
    # st.markdown(hide_menu_style, unsafe_allow_html=True)

    # define streamlit colums
    col1, col2 = st.columns([4, 1], gap="large")

    # build the streamlit sidebar
    build_sidebar()

    # build the main app ui
    build_form(col1, col2)

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "query" not in st.session_state:
        st.session_state["query"] = []

    if "query_text" not in st.session_state:
        st.session_state["query_text"] = []

    # get the users query
    get_text(col1)
    user_input = st.session_state["query"]

    if user_input:
        try:
            output = chain.run(query=user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
        except Exception as exc:
            st.session_state.past.append(user_input)
            st.session_state.generated.append(
                "The datasource does not contain information to answer this question."
            )

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

        st.subheader("Amazon SageMaker Studio")
        st.write(
            """
        [Amazon SageMaker Studio](https://aws.amazon.com/sagemaker/studio/) is a fully integrated development environment (IDE) where you can perform all machine learning (ML) development steps, from preparing data to building, training, and deploying your ML models, inclduing [JumpStart Foundation Models](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models.html).
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

        st.subheader("TICKIT Database")
        st.write(
            """
        [TICKIT database](https://docs.aws.amazon.com/redshift/latest/dg/c_sampledb.html) is a small database consists of seven tables: two fact tables and five dimensions. This sample database track sales activity for the fictional TICKIT web site, where users buy and sell tickets online for sporting events, shows, and concerts.
        """
        )


def build_form(col1, col2):
    with col1:
        with st.container():
            st.title("Natural Language Query Demo")
            st.subheader("Ask questions of your data using natural language.")

        with st.container():
            option = st.selectbox(
                "Choose a datasource:",
                (
                    "Sales (Amazon Redshift)",
                    "Inventory (Amazon RDS for PostgreSQL)",
                    "Human Resources (Amazon Aurora MySQL)",
                ),
                label_visibility=st.session_state.visibility,
                disabled=st.session_state.disabled,
            )
            with st.expander("Sample questions"):
                st.text(
                    """
                How many venues are there in state of NY?
                Which venue hosted the most events?
                How many unique customers made a purchase in May 2022?
                What were the total sales in September 2022?
                What are the largest 3 events based on all time sales? How many tickets were sold to those events?
                Who are the top 5 seller, based on all time sales?
                Who are the top 3 seller, by names, based on all time sales? What were their sales?
                What ingredients are in a chocolate cake?
                Who is the current leader of Canada?
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
