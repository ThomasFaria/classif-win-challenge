from src.utils.data import get_file_system, truncate_txt
from src.db_building.document_chunker import chunk_documents
import pandas as pd
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

EMBEDDING_MODEL = "OrdalieTech/Solon-embeddings-large-0.1"
URL_IN = "s3://projet-dedup-oja/challenge_classification/raw-data/wi_labels.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
CHROMA_DB_LOCAL_DIRECTORY = "data/chroma_db"
COLLECTION_NAME = "test"
TRUNCATE = True


fs = get_file_system()

with fs.open(URL_IN) as f:
    data = pd.read_csv(f, dtype=str)


if TRUNCATE:
    data.loc[:, "description"] = data["description"].apply(
        lambda x: truncate_txt(
            x, [".\nTasks include", "Examples of the occupations classified here:"]
        )
    )

all_splits = chunk_documents(data=data, hf_tokenizer_name=EMBEDDING_MODEL)
emb_model = HuggingFaceEmbeddings(  # load from sentence transformers
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
    show_progress=True,
)
db = Chroma.from_documents(
    collection_name=COLLECTION_NAME,
    documents=all_splits,
    persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
    embedding=emb_model,
    client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
)


print(db.similarity_search("Senior IT Support Engineers", k=10))
