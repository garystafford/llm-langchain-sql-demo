# Natural Language Query (NLQ) Demo

Natural Language Query (NLQ) Demo of Amazon Redshift using OpenAI's LLMs, LangChain, and Streamlit.

Author: Gary A. Stafford

Date: 2023-06-14

License: MIT

## Install Required Packages for `app.py`

```sh
# Required for ChromaDB with Amazon SageMaker Jumpstart
sudo yum update -y && sudo yum install gcc gcc-c++ make -y

# Older version required for sqlalchemy-redshift
python3 -m pip install SQLAlchemy==1.4.48 -q

python3 -m pip install langchain openai sqlalchemy-redshift -Uq
python3 -m pip install streamlit streamlit-chat psycopg2-binary chromadb python-dotenv -Uq
python3 -m pip install pyyaml -q
python3 -m pip install sentence-transformers -Uq --no-cache-dir #--force-reinstall

# Check package versions
pip list | grep "langchain\|openai\|sentence-transformers\|SQLAlchemy"
```

## Start Applications

```
streamlit run app.py --server.runOnSave true
```
