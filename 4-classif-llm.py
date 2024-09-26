import argparse
import os

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from datasets import Dataset
from langchain_core.output_parsers import PydanticOutputParser  # , StrOutputParser
from tqdm import tqdm

from src.constants.llm import (
    BATCH_SIZE,
    DO_SAMPLE,
    LLM_MODEL,
    MAX_NEW_TOKEN,
    TEMPERATURE,
)
from src.constants.paths import URL_DATASET_PREDICTED, URL_DATASET_PROMPTS, URL_LABELS
from src.constants.utils import DEVICE
from src.llm.build_llm import build_llm_model
from src.response.response_llm import LLMResponse, process_response
from src.utils.data import get_file_system


def main(languages: list, quarter: int = None):
    parser = PydanticOutputParser(pydantic_object=LLMResponse)
    fs = get_file_system()

    with fs.open(URL_LABELS) as f:
        labels = pd.read_csv(f, dtype=str)

    generation_args = {
        "max_new_tokens": MAX_NEW_TOKEN,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
    }

    llm, tokenizer = build_llm_model(
        model_name=LLM_MODEL,
        hf_token=os.getenv("HF_TOKEN"),
    )
    for lang in languages:
        print(f"Processing for language: {lang}")

        # Load the dataset
        data = (
            ds.dataset(
                URL_DATASET_PROMPTS.replace("s3://", ""),
                partitioning=["lang"],
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

        if quarter is not None:
            idx_for_subset = [
                ((data.shape[0] // 4) * (quarter - 1)),
                ((data.shape[0] // 4) * quarter),
            ]
            idx_for_subset[-1] = idx_for_subset[-1] if quarter != 4 else data.shape[0]
            data = data.iloc[idx_for_subset[0] : idx_for_subset[1]]

        # Reformat partionnning column
        data["lang"] = data["lang"].str.replace("lang=", "")

        dataset = Dataset.from_dict(data)

        responses = []
        for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
            batch_prompts = dataset[i : i + BATCH_SIZE]["prompt"]
            inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(DEVICE)

            # Generate the output
            outputs = llm.generate(
                **inputs,
                **generation_args,
                pad_token_id=tokenizer.eos_token_id,  # Use the eos_token_id
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            responses.extend(response)

        data.loc[:, "raw_responses"] = responses

        results = []
        for row in tqdm(data.itertuples(), total=data.shape[0]):
            result = process_response(row, parser, labels)
            results.append(result)

        pq.write_to_dataset(
            pa.Table.from_pylist(results),
            root_path=URL_DATASET_PREDICTED,
            partition_cols=["lang", "codable"],
            basename_template=f"part-{{i}}{f'-{quarter}' if quarter else ""}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            filesystem=fs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify function")

    # Add argument languages
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="List of source languages you want to classify",
    )

    parser.add_argument(
        "--quarter",
        type=int,
        required=False,
        help="Quarter of the dataset to process",
    )
    # Parse the command-line arguments
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(languages=args.languages, quarter=args.quarter)
