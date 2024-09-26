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
    DO_SAMPLE,
    LLM_MODEL,
    MAX_NEW_TOKEN,
    TEMPERATURE,
)
from src.constants.paths import URL_DATASET_TRANSLATED, URL_DATASET_WITH_LANG
from src.constants.translation import BATCH_SIZE_TRANSLATION
from src.constants.utils import DEVICE
from src.llm.build_llm import build_llm_model
from src.prompting.prompts import (
    create_translation_prompt,
)
from src.response.response_llm import TranslatorResponse, process_translation
from src.utils.data import get_file_system
from src.utils.mapping import lang_mapping


def main(title_column: str, description_column: str, list_country: list):
    """
    Translate the dataset from all languages to English
    """
    fs = get_file_system()
    parser = PydanticOutputParser(pydantic_object=TranslatorResponse)

    generation_args = {
        "max_new_tokens": MAX_NEW_TOKEN,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
    }

    llm, tokenizer = build_llm_model(
        model_name=LLM_MODEL,
        hf_token=os.getenv("HF_TOKEN"),
    )

    country_map = lang_mapping.loc[lang_mapping["lang_iso_2"].isin(list_country)]
    for lang_iso_2, lang in zip(country_map.lang_iso_2, country_map.lang):
        data = (
            ds.dataset(
                URL_DATASET_WITH_LANG.replace("s3://", ""),
                partitioning=["lang"],
                format="parquet",
                filesystem=fs,
            )
            .to_table()
            .filter((ds.field("lang") == f"lang={lang_iso_2}"))
            .to_pandas()
        )

        if data.empty:
            print(f"No data found for language {lang}. Skipping...")
            continue

        # Reformat partionnning column
        data.loc[:, "lang"] = data.loc[:, "lang"].str.replace("lang=", "")

        if lang_iso_2 in ["en", "un"]:
            # We do not perform translation when text is in english or undefined (bad detected score)
            data.loc[:, f"{title_column}_en"] = data[title_column]
            data.loc[:, f"{description_column}_en"] = data[description_column]
        else:
            print(f"Translating texts from {lang} to English")

            dataset = Dataset.from_dict(data)

            for col in [title_column, description_column]:
                dataset = dataset.map(
                    lambda batch: {
                        f"prompt_{col}": create_translation_prompt(
                            batch, col, parser, lang, description_column=description_column
                        )
                    },
                    batched=False,
                )
                translations = []
                for i in tqdm(range(0, len(dataset), BATCH_SIZE_TRANSLATION)):
                    batch_prompts = dataset[i : i + BATCH_SIZE_TRANSLATION][f"prompt_{col}"]
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
                    translations.extend(response)

                data.loc[:, f"{col}_en"] = translations

            results_title = []
            results_desc = []
            for row in data.itertuples():
                result_title = process_translation(row, f"{title_column}_en", parser)
                result_desc = process_translation(row, f"{description_column}_en", parser)
                results_title.append(result_title)
                results_desc.append(result_desc)

            translation = (
                pd.DataFrame(results_title)
                .rename(columns={"translation": f"{title_column}_en"})
                .merge(
                    pd.DataFrame(results_desc).rename(
                        columns={"translation": f"{description_column}_en"}
                    ),
                    on="id",
                )
            )

            data.loc[:, [f"{title_column}_en", f"{description_column}_en"]] = translation.loc[
                :, [f"{title_column}_en", f"{description_column}_en"]
            ]

        pq.write_to_dataset(
            pa.Table.from_pandas(data),
            root_path=URL_DATASET_TRANSLATED,
            partition_cols=["lang"],
            basename_template="part-{i}.parquet",
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

    # Parse the command-line arguments
    args = parser.parse_args()

    # TITLE_COLUMN = "title_clean"
    # DESCRIPTION_COLUMN = "description_truncated"

    # Call the main function with parsed arguments
    main(args.title_col, args.description_col, args.languages)
