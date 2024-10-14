import argparse

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
    MAX_NEW_TOKEN,
    TEMPERATURE,
)
from src.constants.paths import URL_DATASET_PREDICTED, URL_DATASET_PROMPTS, URL_LABELS
from src.llm.build_llm import cache_model_from_hf_hub
from src.response.response_llm import LLMResponse, process_response
from src.utils.data import get_file_system


def main(languages: list, quarter: int = None):
    # Initialize the output parser for LLM responses
    parser = PydanticOutputParser(pydantic_object=LLMResponse)
    
    # Set up the file system for handling datasets (e.g., S3 or local)
    fs = get_file_system()

    # Cache the specified language model from HuggingFace Hub
    cache_model_from_hf_hub(
        LLM_MODEL,
    )

    # Load the labels dataset
    with fs.open(URL_LABELS) as f:
        labels = pd.read_csv(f, dtype=str)

    # Define sampling parameters for the LLM's generation process
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKEN,  # Set the maximum number of tokens to generate
        temperature=TEMPERATURE,  # Adjust the randomness of the model's predictions
        top_p=0.8,  # Nucleus sampling parameter
        repetition_penalty=1.05,  # Penalize repetition in generated tokens
    )

    # Initialize the language model (LLM) with memory and GPU utilization settings
    llm = LLM(model=LLM_MODEL, max_model_len=20000, gpu_memory_utilization=0.95)

    # Load the dataset containing pre-generated prompts
    data = (
        ds.dataset(
            URL_DATASET_PROMPTS.replace("s3://", ""),  # Replace S3 URL prefix for local access
            partitioning=["lang", "job_desc_extracted"],  # Partition dataset by language and job description
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .filter(ds.field("lang").isin([f"lang={lang}" for lang in languages]))  # Filter data by languages
        .to_pandas()  # Convert PyArrow table to pandas DataFrame
    )

    # If no data is found for the specified languages, return
    if data.empty:
        print(f"No data found for languages {', '.join(languages)}. Skipping...")
        return None

    # If a quarter is specified, process only a subset of the data
    if quarter is not None:
        idx_for_subset = [
            ((data.shape[0] // 3) * (quarter - 1)),  # Determine start index for the subset
            ((data.shape[0] // 3) * quarter),  # Determine end index for the subset
        ]
        idx_for_subset[-1] = idx_for_subset[-1] if quarter != 3 else data.shape[0]
        data = data.iloc[idx_for_subset[0] : idx_for_subset[1]]  # Select the subset of rows

    # Clean up the language and job description columns by removing prefixes
    data["lang"] = data["lang"].str.replace("lang=", "")
    data["job_desc_extracted"] = data["job_desc_extracted"].str.replace("job_desc_extracted=", "")

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

    # Write the updated dataset (with predictions) to a Parquet file
    pq.write_to_dataset(
        pa.Table.from_pandas(data),
        root_path=URL_DATASET_PREDICTED,  # Output directory for predicted data
        partition_cols=["lang", "job_desc_extracted", "codable"],  # Partition by language and job description
        basename_template=f"part-{{i}}{f'-{quarter}' if quarter else ''}.parquet",  # Naming pattern for files
        existing_data_behavior="overwrite_or_ignore",  # Overwrite existing data or ignore
        filesystem=fs,
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

    # Optional argument to specify the dataset quarter to process
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
