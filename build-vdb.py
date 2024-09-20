from src.utils.data import get_file_system, truncate_txt
from src.vector_db.document_chunker import chunk_documents
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from src.constants.paths import URL_LABELS, CHROMA_DB_LOCAL_DIRECTORY
from src.constants.utils import DEVICE
from src.constants.vector_db import EMBEDDING_MODEL, TRUNCATE_LABELS_DESCRIPTION, COLLECTION_NAME


fs = get_file_system()

with fs.open(URL_LABELS) as f:
    data = pd.read_csv(f, dtype=str)


if TRUNCATE_LABELS_DESCRIPTION:
    data.loc[:, "description"] = data["description"].apply(
        lambda x: truncate_txt(
            x, [".\nTasks include", "Examples of the occupations classified here:"]
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
