import argparse
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from langchain_core.output_parsers import PydanticOutputParser  # , StrOutputParser
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams

from src.constants.llm import (
    LLM_MODEL,
    TEMPERATURE,
)
from src.constants.paths import (
    URL_DATASET_PREDICTED,
    URL_DATASET_PREDICTED_FINAL,
    URL_LABELS,
    URL_SUBMISSIONS,
)
from src.response.response_llm import LLMResponse, process_response
from src.utils.data import get_file_system
from src.llm.build_llm import cache_model_from_hf_hub


def main(max_retry: int):
    parser = PydanticOutputParser(pydantic_object=LLMResponse)
    fs = get_file_system()

    cache_model_from_hf_hub(
        LLM_MODEL,
    )

    with fs.open(URL_LABELS) as f:
        labels = pd.read_csv(f, dtype=str)

    sampling_params = SamplingParams(max_tokens=8192, temperature=TEMPERATURE, top_p=0.95)

    llm = LLM(
        model=LLM_MODEL, tokenizer_mode="mistral", config_format="mistral", load_format="mistral"
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
    newly_coded = pd.DataFrame(columns=data.columns)

    for it in range(1, max_retry + 1):
        print(f"Retrying classification for the {it} time")
        batch_prompts = data_uncoded.loc[:, "prompt"].tolist()

        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        responses = [outputs[i].outputs[0].text for i in range(len(outputs))]

        data_uncoded.loc[:, "raw_responses"] = responses

        results = []
        for row in tqdm(data_uncoded.itertuples(), total=data_uncoded.shape[0]):
            result = process_response(row, parser, labels)
            results.append(result)

        results_df = pd.DataFrame(results)
        results_df["codable"] = results_df["codable"].astype(str).str.lower()

        newly_coded = pd.concat([newly_coded] + [results_df[results_df["codable"] == "true"]])
        data_uncoded = results_df[results_df["codable"] == "false"]

    # Concatenate the newly coded data with the already predicted data + still uncoded data
    data_final = pd.concat([data[~data["id"].isin(newly_coded["id"].to_list())]] + [newly_coded])

    pq.write_to_dataset(
        pa.Table.from_pandas(data_final),
        root_path=URL_DATASET_PREDICTED_FINAL,
        partition_cols=["lang", "codable"],
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
