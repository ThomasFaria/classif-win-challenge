import argparse
import os

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from datasets import Dataset
from langchain_core.output_parsers import PydanticOutputParser  # , StrOutputParser
from tqdm import tqdm
from datetime import datetime

from src.constants.llm import (
    BATCH_SIZE,
    DO_SAMPLE,
    LLM_MODEL,
    MAX_NEW_TOKEN,
    TEMPERATURE,
)
from src.constants.paths import (
    URL_DATASET_PREDICTED,
    URL_DATASET_PROMPTS,
    URL_LABELS,
    URL_DATASET_PREDICTED_FINAL,
    URL_SUBMISSIONS,
)
from src.constants.utils import DEVICE
from src.llm.build_llm import build_llm_model
from src.response.response_llm import LLMResponse, process_response
from src.utils.data import get_file_system


def main(max_retry: int):
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
        device=DEVICE,
    )

    # Load the dataset
    data = (
        ds.dataset(
            URL_DATASET_PREDICTED.replace("s3://", ""),
            partitioning=["lang", "codable"],
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .to_pandas()
    )

    # Reformat partionnning column
    data["lang"] = data["lang"].str.replace("lang=", "")
    data["codable"] = data["codable"].str.replace("codable=", "")

    data_uncoded = data[data["codable"] == "false"].copy()

    # Make sure there is no None prompt (TODO: to fix)
    prompts = (
        ds.dataset(
            URL_DATASET_PROMPTS.replace("s3://", ""),
            partitioning=["lang"],
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .to_pandas()
    )
    data_uncoded = pd.merge(
        data_uncoded, prompts.loc[:, ["id", "prompt"]], on="id", how="inner", suffixes=("_left", "")
    ).drop("prompt_left", axis=1)

    for i in range(1, max_retry + 1):
        print(f"Retrying classification for the {i} time")
        dataset = Dataset.from_dict(data_uncoded)

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

        data_uncoded.loc[:, "raw_responses"] = responses

        results = []
        for row in tqdm(data_uncoded.itertuples(), total=data_uncoded.shape[0]):
            result = process_response(row, parser, labels)
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df["codable"] = results_df["codable"].astype(str).str.lower()

        # We write back the new coded in the original dataset
        cols = data.columns
        data = pd.merge(data, results_df, on="id", how="left", suffixes=("", "_new"))

        # Now update the values in the large DataFrame with the new values from the small DataFrame
        for col in results_df.columns:
            if col != "id":  # Skip the 'id' column, we don't want to update it
                data.loc[:, col] = data[col + "_new"].combine_first(data[col])
        data = data[cols]
        data_uncoded = results_df[results_df["codable"] == "false"]

    pq.write_to_dataset(
        pa.Table.from_pandas(data),
        root_path=URL_DATASET_PREDICTED_FINAL,
        partition_cols=["lang", "codable"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )

    with fs.open(
        f"{URL_SUBMISSIONS}/{datetime.now().strftime('%Y-%m-%d-%H-%M')}/classification.csv", "w"
    ) as f:
        data[data["codable"] == "true"].loc[:, ["id", "class_code"]].to_csv(
            f, header=False, index=False
        )


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
