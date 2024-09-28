import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from src.constants.llm import LLM_MODEL
from src.constants.paths import (
    CHROMA_DB_LOCAL_DIRECTORY,
    URL_DATASET_PROMPTS,
    URL_DATASET_TRANSLATED,
    URL_LABELS,
)
from src.constants.utils import DEVICE
from src.constants.vector_db import (
    EMBEDDING_MODEL,
    MAX_CODE_RETRIEVED,
    SEARCH_ALGO,
    TRUNCATE_LABELS_DESCRIPTION,
)
from src.prompting.prompts import create_prompt_with_docs
from src.response.response_llm import LLMResponse
from src.utils.data import extract_info, get_file_system
from src.vector_db.document_chunker import chunk_documents


def main(title_column: str, description_column: str, languages: list):
    fs = get_file_system()
    parser = PydanticOutputParser(pydantic_object=LLMResponse)

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    emb_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )

    with fs.open(URL_LABELS) as f:
        labels_en = pd.read_csv(f, dtype=str)

    if TRUNCATE_LABELS_DESCRIPTION:
        labels_en.loc[:, "description"] = labels_en["description"].apply(lambda x: extract_info(x))

    all_splits = chunk_documents(data=labels_en, hf_tokenizer_name=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        collection_name="labels_embeddings",
        documents=all_splits,
        persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
        embedding=emb_model,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )

    retriever = db.as_retriever(search_type=SEARCH_ALGO, search_kwargs={"k": MAX_CODE_RETRIEVED})

    for lang in languages:
        print(f"Creating prompts for language: {lang}")

        # Load the dataset
        data = (
            ds.dataset(
                URL_DATASET_TRANSLATED.replace("s3://", ""),
                partitioning=["lang", "job_desc_extracted"],
                format="parquet",
                filesystem=fs,
            )
            .to_table()
            .filter((ds.field("lang") == f"lang={lang}"))
            .to_pandas()
        )

        if data.empty:
            print(f"No data found for language {lang}. Skipping...")
            continue

        data["lang"] = data["lang"].str.replace("lang=", "")
        data["job_desc_extracted"] = data["job_desc_extracted"].str.replace("job_desc_extracted=", "")

        prompts = [
            create_prompt_with_docs(
                row,
                parser,
                retriever,
                labels_en,
                **{
                    "description_column": description_column,
                    "title_column": title_column,
                },
            )
            for row in data.itertuples()
        ]

        batch_prompts = tokenizer.apply_chat_template(
            prompts, tokenize=False, add_generation_prompt=True
        )

        # Merge results back to the original dataset
        data.loc[:, "prompt"] = batch_prompts

        pq.write_to_dataset(
            pa.Table.from_pandas(data),
            root_path=URL_DATASET_PROMPTS,
            partition_cols=["lang", "job_desc_extracted"],
            basename_template="part-{i}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            filesystem=fs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompting function")

    # Add arguments for title, description, and languages
    parser.add_argument(
        "--title_col", type=str, required=True, help="Title column you want to use for prompts"
    )
    parser.add_argument(
        "--description_col",
        type=str,
        required=True,
        help="Description column you want to use for prompts",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="List of source languages for which you want to create prompts",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.title_col, args.description_col, args.languages)
