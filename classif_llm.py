from src.utils.data import get_file_system
from src.llm.build_llm import build_llm_model
import torch

import os
from src.prompting.prompts import RAG_PROMPT_TEMPLATE, format_docs
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pyarrow.dataset as ds

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


EMBEDDING_MODEL = "OrdalieTech/Solon-embeddings-large-0.1"
URL_IN = "s3://projet-dedup-oja/challenge_classification/raw-data/wi_labels.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
CHROMA_DB_LOCAL_DIRECTORY = "data/chroma_db"
COLLECTION_NAME = "test"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


fs = get_file_system()

data = (
    ds.dataset(
        URL_IN,
        partitioning=["lang"],
        format="parquet",
        filesystem=fs,
    )
    .to_table()
    .filter((ds.field("lang") == "lang=fr"))
    .to_pandas()
)

llm, tokenizer = build_llm_model(
    model_name=LLM_MODEL,
    quantization_config=True,
    config=True,
    token=os.getenv("HF_TOKEN"),
)


emb_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    multi_process=False,
    model_kwargs={"device": device, "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
    show_progress=True,
)
db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
    embedding_function=emb_model,
)


retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 30})

description = data.loc[0, "description"]
title = data.loc[0, "title"]

# Ici faut voir ce qu'on fait mais faut virer/nettoyer/selectionner les documents
retrieved_docs = retriever.invoke(" ".join([title, description]))
retrieved_docs_unique = []
for item in retrieved_docs:
    if item not in retrieved_docs_unique:
        retrieved_docs_unique.append(item)

prompt_template = PromptTemplate.from_template(
    template=RAG_PROMPT_TEMPLATE,
)

# TODO: make sure the prompt is not to long
prompt = prompt_template.format(
    **{
        "title": title,
        "description": description,
        "proposed_categories": format_docs(retrieved_docs_unique),
    }
)

# On crée la chaine de traitement
chain = prompt_template | llm | StrOutputParser()


# On interroge le LLM
chain.invoke(
    {
        "title": title,
        "description": description,
        "proposed_categories": format_docs(retrieved_docs_unique),
    },
)
