import pandas as pd
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.constants.paths import CHROMA_DB_LOCAL_DIRECTORY, URL_LABELS
from src.constants.utils import DEVICE
from src.constants.vector_db import COLLECTION_NAME, EMBEDDING_MODEL, TRUNCATE_LABELS_DESCRIPTION
from src.utils.data import get_file_system, truncate_txt
from src.vector_db.document_chunker import chunk_documents

fs = get_file_system()

with fs.open(URL_LABELS) as f:
    data = pd.read_csv(f, dtype=str)


if TRUNCATE_LABELS_DESCRIPTION:
    data.loc[:, "description"] = data["description"].apply(
        lambda x: truncate_txt(
            x,
            [
                ".\nTasks include",
                "Examples of the occupations classified here:",
            ],  # Find other sentence to truncate
        )
    )

all_splits = chunk_documents(data=data, hf_tokenizer_name=EMBEDDING_MODEL)

emb_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
    show_progress=False,
)
db = Chroma.from_documents(
    collection_name=COLLECTION_NAME,
    documents=all_splits,
    persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
    embedding=emb_model,
    client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
)

print("Vector DB is built")
