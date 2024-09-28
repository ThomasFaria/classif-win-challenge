import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from langchain_core.output_parsers import PydanticOutputParser  # , StrOutputParser
from tqdm import tqdm

from src.constants.llm import (
    LLM_MODEL,
    MAX_NEW_TOKEN,
    TEMPERATURE,
)
from src.constants.paths import URL_DATASET_TRANSLATED, URL_DATASET_WITH_LANG
from src.prompting.prompts import (
    create_translation_prompt,
)
from src.response.response_llm import TranslatorResponse, process_translation
from src.utils.data import get_file_system
from vllm import LLM
from vllm.sampling_params import SamplingParams
from src.llm.build_llm import cache_model_from_hf_hub


def main(title_column: str, description_column: str, languages: list, quarter: int = None):
    """
    Translate the dataset from all languages to English
    """
    fs = get_file_system()
    parser = PydanticOutputParser(pydantic_object=TranslatorResponse)

    cache_model_from_hf_hub(
        LLM_MODEL,
    )

    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKEN, temperature=TEMPERATURE, top_p=0.8, repetition_penalty=1.05)

    llm = LLM(
        model=LLM_MODEL,
    )

    for lang in languages:
        data = (
            ds.dataset(
                URL_DATASET_WITH_LANG.replace("s3://", ""),
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
        data.loc[:, "lang"] = data.loc[:, "lang"].str.replace("lang=", "")

        print(f"Translating texts from {lang} to English")

        # Create the prompt
        batch_prompts = [
            create_translation_prompt(
                row,
                parser,
                **{
                    "description_column": description_column,
                    "title_column": title_column,
                },
            )
            for row in data.itertuples()
        ]

        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        translations = [outputs[i].outputs[0].text for i in range(len(outputs))]

        data.loc[:, "raw_responses"] = translations

        results = []
        for row in tqdm(data.itertuples(), total=data.shape[0]):
            result = process_translation(row, parser)
            results.append(result)

        data = data.merge(
            pd.DataFrame(results).rename(
                columns={"description": "description_en", "title": "title_en"}
            ),
            on="id",
        )

        pq.write_to_dataset(
            pa.Table.from_pandas(data),
            root_path=URL_DATASET_TRANSLATED,
            partition_cols=["lang"],
            basename_template=f"part-{{i}}{f'-{quarter}' if quarter else ""}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            filesystem=fs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translation function")

    # Add arguments for title, description, and languages
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

    parser.add_argument(
        "--quarter",
        type=int,
        required=False,
        help="Quarter of the dataset to process",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        title_column=args.title_col,
        description_column=args.description_col,
        languages=args.languages,
        quarter=args.quarter,
    )
