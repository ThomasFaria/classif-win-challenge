import argparse
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser  # , StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams

from src.constants.llm import (
    LLM_MODEL,
    MAX_NEW_TOKEN,
    TEMPERATURE,
)
from src.constants.paths import (
    CHROMA_DB_LOCAL_DIRECTORY,
    URL_DATASET_PREDICTED,
    URL_DATASET_PREDICTED_FINAL,
    URL_LABELS,
    URL_SUBMISSIONS,
)
from src.constants.utils import DEVICE
from src.constants.vector_db import (
    EMBEDDING_MODEL,
    SEARCH_ALGO,
    TRUNCATE_LABELS_DESCRIPTION,
)
from src.llm.build_llm import cache_model_from_hf_hub
from src.prompting.prompts import create_prompt_with_docs
from src.response.response_llm import LLMResponse, process_response
from src.utils.data import extract_info, get_file_system
from src.vector_db.document_chunker import chunk_documents


def main(max_retry: int):
    parser = PydanticOutputParser(pydantic_object=LLMResponse)
    fs = get_file_system()

    cache_model_from_hf_hub(
        LLM_MODEL,
    )

    emb_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )

    with fs.open(URL_LABELS) as f:
        labels_en = pd.read_csv(f, dtype=str)

    if TRUNCATE_LABELS_DESCRIPTION:
        labels_en.loc[:, "description"] = labels_en["description"].apply(
            lambda x: extract_info(x, only_description=True)
        )

    all_splits = chunk_documents(data=labels_en, hf_tokenizer_name=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        collection_name="labels_embeddings",
        documents=all_splits,
        persist_directory=CHROMA_DB_LOCAL_DIRECTORY,
        embedding=emb_model,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )

    retriever = db.as_retriever(search_type=SEARCH_ALGO, search_kwargs={"k": 100})

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKEN, temperature=TEMPERATURE, top_p=0.8, repetition_penalty=1.05
    )
    llm = LLM(model=LLM_MODEL, max_model_len=20000, gpu_memory_utilization=0.95)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    # Load the dataset
    data = (
        ds.dataset(
            URL_DATASET_PREDICTED.replace("s3://", ""),
            partitioning=["lang", "job_desc_extracted", "codable"],
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .to_pandas()
    )

    # Reformat partionnning column
    data["lang"] = data["lang"].str.replace("lang=", "")
    data["job_desc_extracted"] = data["job_desc_extracted"].str.replace("job_desc_extracted=", "")
    data["codable"] = data["codable"].str.replace("codable=", "")

    data_uncoded = data[data["codable"] == "false"].copy()
    newly_coded = pd.DataFrame(columns=data.columns)

    prompts = [
        create_prompt_with_docs(
            row,
            parser,
            retriever,
            labels_en,
            **{
                "description_column": "description_en",
                "title_column": "title_en",
            },
        )
        for row in data_uncoded.itertuples()
    ]

    batch_prompts = tokenizer.apply_chat_template(
        prompts, tokenize=False, add_generation_prompt=True
    )
    data_uncoded.loc[:, "prompt"] = batch_prompts

    for it in range(1, max_retry + 1):
        print(f"Retrying classification for the {it} time")
        batch_prompts = data_uncoded.loc[:, "prompt"].tolist()

        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        responses = [outputs[i].outputs[0].text for i in range(len(outputs))]

        data_uncoded.loc[:, "raw_responses"] = responses

        results = []
        for row in tqdm(data_uncoded.itertuples(), total=data_uncoded.shape[0]):
            result = process_response(row, parser, labels_en)
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df["codable"] = results_df["codable"].astype(str).str.lower()

        newly_coded = pd.concat([newly_coded] + [results_df[results_df["codable"] == "true"]])
        newly_coded = newly_coded.merge(data_uncoded, on="id")
        data_uncoded = results_df[results_df["codable"] == "false"].merge(
            data_uncoded.loc[:, ["id", "prompt"]], on="id"
        )

    # Concatenate the newly coded data with the already predicted data + still uncoded data
    data_final = pd.concat([data[~data["id"].isin(newly_coded["id"].to_list())]] + [newly_coded])

    pq.write_to_dataset(
        pa.Table.from_pandas(data_final),
        root_path=URL_DATASET_PREDICTED_FINAL,
        partition_cols=["lang", "job_desc_extracted", "codable"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )

    assert data_final.shape[0] == 25665

    with fs.open(
        f"{URL_SUBMISSIONS}/{datetime.now().strftime('%Y-%m-%d-%H-%M')}/classification.csv", "w"
    ) as f:
        submissions = data_final.loc[:, ["id", "class_code", "codable"]]
        submissions.loc[submissions["codable"] == "false", "class_code"] = None
        submissions.loc[:, ["id", "class_code"]].fillna("0110").to_csv(f, header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retry classification")

    # Add argument languages
    parser.add_argument(
        "--max_retry",
        type=int,
        required=True,
        help="Max retry",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.max_retry)
