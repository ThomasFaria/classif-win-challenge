import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents.base import Document
from transformers import AutoTokenizer


def chunk_documents(
    data: pd.DataFrame,
    hf_tokenizer_name: str,
    separators: list = ["\n\n", "\n", ".", " ", ""],
) -> list[Document]:
    """
    Chunks documents from a dataframe into smaller pieces using specified tokenizer settings or custom settings.

    Parameters:
    - data (pd.DataFrame): The dataframe containing documents to be chunked.
    - hf_tokenizer_name (str): Name of the Hugging Face tokenizer to use.
    - separators (list, optional): List of separators to use for splitting the text.

    Returns:
    - List[Document]: A tuple containing the list of processed unique document chunks and chunking information.
    """

    # advantage of using a loader
    # No need to know which metadata are stored in the dataframe
    # Every column except page_content_column contains metadata
    document_list = DataFrameLoader(data, page_content_column="description").load()

    autokenizer, chunk_size, chunk_overlap = compute_autotokenizer_chunk_size(hf_tokenizer_name)

    # Initialize token splitter
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        autokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    # Split documents into chunks
    docs_processed = text_splitter.split_documents(document_list)

    # Remove duplicates
    #    unique_texts = set()
    #    docs_processed_unique = []
    #    for doc in docs_processed:
    #        if doc.page_content not in unique_texts:
    #            unique_texts.add(doc.page_content)
    #            docs_processed_unique.append(doc)

    print(f"Number of created chunks: {len(docs_processed)} in the Vector Database")

    return docs_processed


def compute_autotokenizer_chunk_size(hf_tokenizer_name: str) -> tuple:
    """
    Computes the chunk size and chunk overlap for text processing based on the
    capabilities of a Hugging Face tokenizer.

    Parameters:
    hf_tokenizer_name (str): The name of the Hugging Face tokenizer to use.

    Returns:
    tuple: A tuple containing the tokenizer instance, the chunk size, and the chunk overlap.
    """
    # Load the tokenizer
    autokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)

    # Get the maximum token length the tokenizer can handle
    chunk_size = autokenizer.model_max_length

    # Compute chunk overlap as 10% of the chunk size
    chunk_overlap = int(chunk_size * 0.1)

    return autokenizer, chunk_size, chunk_overlap
