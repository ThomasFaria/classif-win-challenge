from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_retriever(
    embedding_model: str,
    collection_name: str,
    db_directory: str,
    search_algo: str = "similarity",
    max_code_retrieved: int = 10,
    device: str = "auto",
    embedding_kwargs: Optional[dict] = None,
    search_kwargs: Optional[dict] = None,
):
    """
    Load a Chroma retriever using HuggingFace embeddings with specified configuration.

    Args:
        embedding_model (str): Name of the embedding model to use.
        collection_name (str): Name of the Chroma collection to use.
        db_directory (str): Directory path to the Chroma database.
        search_algo (str, optional): Search algorithm to use (default is "similarity").
        max_code_retrieved (int, optional): Maximum number of results to retrieve (default is 10).
        device (str, optional): Device to run the embedding model on (e.g., 'cpu', 'cuda') (default is 'auto').
        embedding_kwargs (dict, optional): Additional keyword arguments for the embedding model.
        search_kwargs (dict, optional): Additional keyword arguments for the search retriever.

    Returns:
        retriever: A Chroma retriever object for searching embeddings.
    """

    # Set default embedding and search arguments if not provided
    embedding_kwargs = embedding_kwargs or {"normalize_embeddings": True}
    search_kwargs = search_kwargs or {"k": max_code_retrieved}

    # Load embedding model
    emb_model = HuggingFaceEmbeddings(
        model_name=embedding_model,
        multi_process=False,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs=embedding_kwargs,
        show_progress=False,
    )

    # Initialize Chroma vector store
    db = Chroma(
        collection_name=collection_name,
        persist_directory=db_directory,
        embedding_function=emb_model,
    )

    # Return the retriever
    return db.as_retriever(search_type=search_algo, search_kwargs=search_kwargs)
