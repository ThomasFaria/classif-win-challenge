import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
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
from src.llm.build_llm import cache_model_from_hf_hub
from src.prompting.prompts import create_prompt_with_docs
from src.response.response_llm import LLMResponse
from src.utils.data import extract_info, get_file_system
from src.vector_db.document_chunker import chunk_documents


def main(
    title_column: str,
    description_column: str,
    languages: list,
    third: int = None,
    use_s3: bool = False,
):
    if use_s3:
        # Get the file system handler to access dataset files
        fs = get_file_system()

        # Load the labels dataset
        with fs.open(URL_LABELS) as f:
            labels = pd.read_csv(f, dtype=str)

        # Cache the embedding model from HuggingFace Hub (need s3 bucket access)
        cache_model_from_hf_hub(EMBEDDING_MODEL)
    else:
        # Load the labels dataset
        labels = pd.read_csv(f"data/{"/".join(URL_LABELS.split("/")[-2:])}", dtype=str)

    # Initialize output parser for LLM responses
    parser = PydanticOutputParser(pydantic_object=LLMResponse)

    # Load the tokenizer for the language model
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    # Create an embedding model for generating document embeddings
    emb_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )

    # Optionally truncate the label descriptions so that to keep only relevant information and not overload the prompt
    if TRUNCATE_LABELS_DESCRIPTION:
        labels.loc[:, "description"] = labels["description"].apply(
            lambda x: extract_info(x, paragraphs=["description", "tasks", "examples"])
        )

    # Chunk the labels into smaller parts for more efficient processing
    all_splits = chunk_documents(data=labels, hf_tokenizer_name=EMBEDDING_MODEL)

    # Initialize a local Chroma vector database with the label embeddings
    db = Chroma.from_documents(
        collection_name="labels_embeddings",
        documents=all_splits,
        persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
        embedding=emb_model,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )

    # Create a retriever for retrieving relevant label embeddings based on a search algorithm
    retriever = db.as_retriever(search_type=SEARCH_ALGO, search_kwargs={"k": MAX_CODE_RETRIEVED})

    # Loop through the provided languages
    for lang in languages:
        print(f"Creating prompts for language: {lang}")

        if use_s3:
            # Load the dataset from Parquet files, filtering by the specified languages
            data = (
                pq.ParquetDataset(
                    URL_DATASET_TRANSLATED.replace("s3://", ""),
                    filters=[("lang", "in", languages)],
                    filesystem=fs,
                )
                .read()
                .to_pandas()
            )
        else:
            # Load the dataset from Parquet files, filtering by the specified languages
            data = (
                pq.ParquetDataset(
                    f"data/{"/".join(URL_DATASET_TRANSLATED.split("/")[-2:])}",
                    filters=[("lang", "in", languages)],
                )
                .read()
                .to_pandas()
            )

        # If there is no data for the current language, skip to the next one
        if data.empty:
            print(f"No data found for language {lang}. Skipping...")
            continue

        # If a specific third is provided, process only a subset of the data
        if third is not None:
            idx_for_subset = [
                ((data.shape[0] // 3) * (third - 1)),
                ((data.shape[0] // 3) * third),
            ]
            idx_for_subset[-1] = idx_for_subset[-1] if third != 3 else data.shape[0]
            data = data.iloc[idx_for_subset[0] : idx_for_subset[1]]

        # Generate prompts for each row of the dataset
        prompts = [
            create_prompt_with_docs(
                row,
                parser,
                retriever,
                **{
                    "description_column": description_column,
                    "title_column": title_column,
                },
            )
            for row in tqdm(data.itertuples())
        ]

        # Apply a chat-based template to the generated prompts
        batch_prompts = tokenizer.apply_chat_template(
            prompts, tokenize=False, add_generation_prompt=True
        )

        # Merge the generated prompts back into the original dataset
        data.loc[:, "prompt"] = batch_prompts

        if use_s3:
            # Write the updated dataset with prompts to a Parquet dataset
            pq.write_to_dataset(
                pa.Table.from_pandas(data),
                root_path=URL_DATASET_PROMPTS,
                partition_cols=["lang", "job_desc_extracted"],
                basename_template=f"part-{{i}}{f'-{third}' if third else ""}.parquet",
                existing_data_behavior="overwrite_or_ignore",
                filesystem=fs,
            )
        else:
            # Write the updated dataset with prompts to a Parquet dataset
            pq.write_to_dataset(
                pa.Table.from_pandas(data),
                root_path=f"data/{'/'.join(URL_DATASET_PROMPTS.split('/')[-2:])}",
                partition_cols=["lang", "job_desc_extracted"],
                basename_template=f"part-{{i}}{f'-{third}' if third else ""}.parquet",
                existing_data_behavior="overwrite_or_ignore",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompting function")

    # Add arguments for title column, description column, and languages
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

    # Optional argument for specifying the dataset third to process
    parser.add_argument(
        "--third",
        type=int,
        required=False,
        help="Third of the dataset to process",
    )

    # Optional argument for specifying if S3 storage should be used
    parser.add_argument(
        "--use_s3",
        type=int,
        choices=[0, 1],
        required=False,
        help="Use S3 storage for reading and writing data",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        title_column=args.title_col,
        description_column=args.description_col,
        languages=args.languages,
        third=args.third,
        use_s3=args.use_s3,
    )
