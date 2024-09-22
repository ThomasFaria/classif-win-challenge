import os

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from langchain_core.output_parsers import PydanticOutputParser  # , StrOutputParser
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from src.constants.llm import (
    DO_SAMPLE,
    LLM_MODEL,
    MAX_NEW_TOKEN,
    RETURN_FULL_TEXT,
    TEMPERATURE,
)
from src.constants.paths import URL_DATASET_TRANSLATED, URL_DATASET_WITH_LANG
from src.constants.translation import BATCH_SIZE_TRANSLATION
from src.llm.build_llm import build_llm_model
from src.prompting.prompts import (
    TRANSLATION_PROMPT_TEMPLATE_DESC,
    TRANSLATION_PROMPT_TEMPLATE_TITLE,
)
from src.response.response_llm import TranslatorResponse, process_translation
from src.utils.data import get_file_system
from src.utils.mapping import lang_mapping
from datasets import Dataset


fs = get_file_system()
parser = PydanticOutputParser(pydantic_object=TranslatorResponse)

generation_args = {
    "max_new_tokens": MAX_NEW_TOKEN,
    "return_full_text": RETURN_FULL_TEXT,
    "do_sample": DO_SAMPLE,
    "temperature": TEMPERATURE,
}

llm, tokenizer = build_llm_model(
    model_name=LLM_MODEL,
    quantization_config=True,
    config=True,
    token=os.getenv("HF_TOKEN"),
    generation_args=generation_args,
)


TITLE_COLUMN = "title_clean"
DESCRIPTION_COLUMN = "description_truncated"

fs = get_file_system()


def create_prompt(batch, col):
    template = (
        TRANSLATION_PROMPT_TEMPLATE_DESC
        if batch[col] == DESCRIPTION_COLUMN
        else TRANSLATION_PROMPT_TEMPLATE_TITLE
    )
    return PromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    ).format(
        **{
            "source_language": lang,
            "txt_to_translate": batch[col],
        }
    )


for lang_iso_2, lang in zip(lang_mapping.lang_iso_2, lang_mapping.lang):
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
        continue

    # Reformat partionnning column
    data.loc[:, "lang"] = data.loc[:, "lang"].str.replace("lang=", "")

    if lang_iso_2 == "en":
        # We do not perform translation when text is in english
        data.loc[:, f"{TITLE_COLUMN}_en"] = data[TITLE_COLUMN]
        data.loc[:, f"{DESCRIPTION_COLUMN}_en"] = data[DESCRIPTION_COLUMN]
    else:
        print(f"Translating texts from {lang} to English")

        dataset = Dataset.from_dict(data)

        for col in [TITLE_COLUMN, DESCRIPTION_COLUMN]:
            dataset = dataset.map(
                lambda batch: {f"prompt_{col}": create_prompt(batch, col)},
                batched=False,  # Disable batching here to process one row at a time for the prompt creation
            )

            translations = []
            for i in tqdm(range(0, len(dataset), BATCH_SIZE_TRANSLATION)):
                batch_prompts = dataset[i : i + BATCH_SIZE_TRANSLATION][f"prompt_{col}"]
                response_batch = llm.generate(
                    prompts=batch_prompts
                )  # Use generate for batch processing
                translations.extend(
                    [gen[0].text for gen in response_batch.generations]
                )  # Extract generated text from each response

            data.loc[:, f"{col}_en"] = translations

        results_title = []
        results_desc = []
        for row in data.itertuples():
            result_title = process_translation(row, f"{TITLE_COLUMN}_en", parser)
            result_desc = process_translation(row, f"{DESCRIPTION_COLUMN}_en", parser)
            results_title.append(result_title)
            results_desc.append(result_desc)

        translation = (
            pd.DataFrame(results_title)
            .rename(columns={"translation": f"{TITLE_COLUMN}_en"})
            .merge(
                pd.DataFrame(results_desc).rename(
                    columns={"translation": f"{DESCRIPTION_COLUMN}_en"}
                ),
                on="id",
            )
        )

        data.loc[:, [f"{TITLE_COLUMN}_en", f"{DESCRIPTION_COLUMN}_en"]] = translation.loc[
            :, [f"{TITLE_COLUMN}_en", f"{DESCRIPTION_COLUMN}_en"]
        ]

    pq.write_to_dataset(
        pa.Table.from_pandas(data),
        root_path=URL_DATASET_TRANSLATED,
        partition_cols=["lang"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )
