import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_core.output_parsers import PydanticOutputParser  # , StrOutputParser
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams

from src.constants.llm import (
    LLM_MODEL,
    MAX_NEW_TOKEN,
    REP_PENALTY,
    TEMPERATURE,
    TOP_P,
)
from src.constants.paths import URL_DATASET_PREDICTED, URL_DATASET_PROMPTS, URL_LABELS
from src.llm.build_llm import cache_model_from_hf_hub
from src.response.response_llm import LLMResponse, process_response
from src.utils.data import get_file_system


def main(languages: list, third: int = None, use_s3: bool = False):
    if use_s3:
        # Get the file system handler to access dataset files
        fs = get_file_system()

        # Load the labels dataset
        with fs.open(URL_LABELS) as f:
            labels = pd.read_csv(f, dtype=str)

        # Cache the LLM model from HuggingFace Hub (need s3 bucket access)
        cache_model_from_hf_hub(LLM_MODEL)
    else:
        # Load the labels dataset
        labels = pd.read_csv(f"data/{"/".join(URL_LABELS.split("/")[-2:])}", dtype=str)

    # Initialize the output parser for LLM responses
    parser = PydanticOutputParser(pydantic_object=LLMResponse)

    # Define sampling parameters for the LLM's generation process
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKEN,  # Set the maximum number of tokens to generate
        temperature=TEMPERATURE,  # Adjust the randomness of the model's predictions
        top_p=TOP_P,  # Nucleus sampling parameter
        repetition_penalty=REP_PENALTY,  # Penalize repetition in generated tokens
    )

    # Initialize the language model (LLM) with memory and GPU utilization settings
    llm = LLM(model=LLM_MODEL, max_model_len=20000, gpu_memory_utilization=0.95)

    if use_s3:
        # Load the dataset from Parquet files, filtering by the specified languages
        data = (
            pq.ParquetDataset(
                URL_DATASET_PROMPTS.replace("s3://", ""),
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
                f"data/{"/".join(URL_DATASET_PROMPTS.split("/")[-2:])}",
                filters=[("lang", "in", languages)],
            )
            .read()
            .to_pandas()
        )

    # If no data is found for the specified languages, return
    if data.empty:
        print(f"No data found for languages {', '.join(languages)}. Skipping...")
        return None

    # If a third is specified, process only a subset of the data
    if third is not None:
        idx_for_subset = [
            ((data.shape[0] // 3) * (third - 1)),  # Determine start index for the subset
            ((data.shape[0] // 3) * third),  # Determine end index for the subset
        ]
        idx_for_subset[-1] = idx_for_subset[-1] if third != 3 else data.shape[0]
        data = data.iloc[idx_for_subset[0] : idx_for_subset[1]]  # Select the subset of rows

    # Get a list of prompts to send to the LLM for generation
    batch_prompts = data.loc[:, "prompt"].tolist()

    # Generate responses from the LLM using the batch prompts and sampling parameters
    outputs = llm.generate(batch_prompts, sampling_params=sampling_params)

    # Extract the raw text responses from the LLM output
    responses = [outputs[i].outputs[0].text for i in range(len(outputs))]

    # Store the raw responses back into the dataset
    data.loc[:, "raw_responses"] = responses

    # Process the LLM responses and store the results
    results = []
    for row in tqdm(data.itertuples(), total=data.shape[0]):  # Use tqdm to show progress
        result = process_response(row, parser, labels)  # Parse and process each response
        results.append(result)

    # Merge the processed results back into the original dataset
    data = data.merge(
        pd.DataFrame(results),  # Convert results to a DataFrame
        on="id",  # Merge based on the 'id' column
    )

    if use_s3:
        # Write the updated dataset (with predictions) to a Parquet file
        pq.write_to_dataset(
            pa.Table.from_pandas(data),
            root_path=URL_DATASET_PREDICTED,  # Output directory for predicted data
            partition_cols=[
                "lang",
                "job_desc_extracted",
                "codable",
            ],  # Partition by language and job description
            basename_template=f"part-{{i}}{f'-{third}' if third else ''}.parquet",  # Naming pattern for files
            existing_data_behavior="overwrite_or_ignore",  # Overwrite existing data or ignore
            filesystem=fs,
        )
    else:
        # Write the updated dataset (with predictions) to a Parquet file
        pq.write_to_dataset(
            pa.Table.from_pandas(data),
            root_path=f"data/{'/'.join(URL_DATASET_PREDICTED.split('/')[-2:])}",  # Output directory for predicted data
            partition_cols=[
                "lang",
                "job_desc_extracted",
                "codable",
            ],  # Partition by language and job description
            basename_template=f"part-{{i}}{f'-{third}' if third else ''}.parquet",  # Naming pattern for files
            existing_data_behavior="overwrite_or_ignore",  # Overwrite existing data or ignore
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify function")

    # Add argument for languages
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="List of source languages you want to classify",
    )

    # Optional argument to specify the dataset third to process
    parser.add_argument(
        "--third",
        type=int,
        required=False,
        help="Third of the dataset to process",
    )

    # Optional argument for specifying if S3 storage should be used
    parser.add_argument(
        "--use_s3",
        type=bool,
        required=False,
        help="Use S3 storage for reading and writing data",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(languages=args.languages, third=args.third)
