import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from langchain_core.output_parsers import PydanticOutputParser  # For parsing LLM outputs
from tqdm import tqdm  # For progress bar during iteration
from transformers import AutoTokenizer  # Tokenizer for text processing
from vllm import LLM  # LLM for language generation
from vllm.sampling_params import SamplingParams  # Parameters for controlling text generation

from src.constants.llm import (
    LLM_MODEL,
    MAX_NEW_TOKEN,
    TEMPERATURE,
)
from src.constants.paths import URL_DATASET_TRANSLATED, URL_DATASET_WITH_LANG
from src.llm.build_llm import cache_model_from_hf_hub  # Cache the model from Hugging Face Hub
from src.prompting.prompts import (
    create_translation_prompt,  # Function to create translation prompts
)
from src.response.response_llm import (  # Response handling and translation processing
    TranslatorResponse,
    process_translation,
)
from src.utils.data import get_file_system  # Utility for file system operations


def main(
    title_column: str,
    description_column: str,
    languages: list,
    quarter: int = None,
    use_s3: bool = False,
):
    """
    Main function to translate job descriptions from various languages into English.

    Args:
        title_column (str): The column name that holds job titles.
        description_column (str): The column name that holds job descriptions.
        languages (list): A list of languages to translate from.
        quarter (int, optional): The quarter of the dataset to process (used for partitioning).
    """

    if use_s3:
        # Get the file system handler to access dataset files
        fs = get_file_system()

        # Load the dataset from Parquet files, filtering by the specified languages
        data = (
            pq.ParquetDataset(URL_DATASET_WITH_LANG.replace("s3://", ""), filesystem=fs)
            .read()
            .filter(ds.field("lang").isin([f"lang={lang}" for lang in languages]))
            .to_pandas()
        )
        # TODO

        # Cache the pre-trained LLM model locally (from Hugging Face Hub)
        cache_model_from_hf_hub(LLM_MODEL)
    else:
        # Load the dataset from Parquet files, filtering by the specified languages
        data = (
            pq.ParquetDataset(f"data/{"/".join(URL_DATASET_WITH_LANG.split("/")[-2:])}")
            .read()
            .filter(ds.field("lang").isin([f"lang={lang}" for lang in languages]))
            .to_pandas()
        )
        # TODO

    # If no data is found for the specified languages, skip the process
    if data.empty:
        print(f"No data found for languages {', '.join(languages)}. Skipping...")
        return None

    # Initialize an output parser for handling LLM responses related to translation
    parser = PydanticOutputParser(pydantic_object=TranslatorResponse)

    # Define sampling parameters for text generation (controls temperature, max tokens, etc.)
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKEN, temperature=TEMPERATURE, top_p=0.8, repetition_penalty=1.05
    )

    # Load the tokenizer for the LLM model
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    # Initialize the LLM with specific memory and performance settings
    llm = LLM(model=LLM_MODEL, max_model_len=20000, gpu_memory_utilization=0.95)

    # If quarter is specified, process only a subset of the data for that quarter
    if quarter is not None:
        idx_for_subset = [
            ((data.shape[0] // 3) * (quarter - 1)),  # Start index for the subset
            ((data.shape[0] // 3) * quarter),  # End index for the subset
        ]
        idx_for_subset[-1] = (
            idx_for_subset[-1] if quarter != 3 else data.shape[0]
        )  # Adjust for the last quarter
        data = data.iloc[idx_for_subset[0] : idx_for_subset[1]]  # Select subset

    # Generate translation prompts for each row in the dataset
    prompts = [
        create_translation_prompt(
            row,
            parser,
            **{
                "description_column": description_column,  # Column containing job descriptions
                "title_column": title_column,  # Column containing job titles
            },
        )
        for row in data.itertuples()  # Iterate through dataset rows
    ]

    # Apply a template to the prompts before sending them to the LLM
    batch_prompts = tokenizer.apply_chat_template(
        prompts, tokenize=False, add_generation_prompt=True
    )

    # Generate translations using the LLM and the specified sampling parameters
    outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
    translations = [
        outputs[i].outputs[0].text for i in range(len(outputs))
    ]  # Extract translations from output

    # Add the raw translations back into the original dataset
    data.loc[:, "raw_translations"] = translations

    # Process the translations and prepare the final results
    results = []
    for row in tqdm(data.itertuples(), total=data.shape[0]):  # Progress bar for iteration
        result = process_translation(row, parser)  # Process each translation response
        results.append(result)

    # Merge the processed translation results into the original dataset
    data = data.merge(
        pd.DataFrame(results).rename(
            columns={"description": "description_en", "title": "title_en"}
        ),
        on="id",  # Merge on the 'id' column
    )

    if use_s3:
        # Write the updated dataset (with translations) to a Parquet dataset
        pq.write_to_dataset(
            pa.Table.from_pandas(data),  # Convert pandas DataFrame back to Arrow Table
            root_path=URL_DATASET_TRANSLATED,  # Save dataset to the translated data path
            partition_cols=[
                "lang",
                "job_desc_extracted",
            ],  # Partition by language and job description
            basename_template=f"part-{{i}}{f'-{quarter}' if quarter else ""}.parquet",  # Filename template for Parquet parts
            existing_data_behavior="overwrite_or_ignore",  # Overwrite or ignore existing data
            filesystem=fs,  # Use the specified file system
        )
    else:
        # Write the updated dataset (with translations) to a Parquet dataset
        pq.write_to_dataset(
            pa.Table.from_pandas(data),  # Convert pandas DataFrame back to Arrow Table
            root_path=f"data/{'/'.join(URL_DATASET_TRANSLATED.split('/')[-2:])}",  # Save dataset to the translated data path
            partition_cols=[
                "lang",
                "job_desc_extracted",
            ],  # Partition by language and job description
            basename_template=f"part-{{i}}{f'-{quarter}' if quarter else ""}.parquet",  # Filename template for Parquet parts
            existing_data_behavior="overwrite_or_ignore",  # Overwrite or ignore existing data
        )


if __name__ == "__main__":
    # Set up argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Translation function")

    # Add arguments for title column, description column, and languages
    parser.add_argument(
        "--title_col", type=str, required=True, help="Title column you want to translate"
    )
    parser.add_argument(
        "--description_col",
        type=str,
        required=True,
        help="Description column you want to translate",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help="List of source languages you want to translate",
    )

    # Optional argument for specifying the quarter of the dataset to process
    parser.add_argument(
        "--quarter",
        type=int,
        required=False,
        help="Quarter of the dataset to process",
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

    # Call the main function with the parsed arguments
    main(
        title_column=args.title_col,
        description_column=args.description_col,
        languages=args.languages,
        quarter=args.quarter,
        use_s3=args.use_s3,
    )
