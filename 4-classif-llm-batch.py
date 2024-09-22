import os

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from langchain_core.output_parsers import PydanticOutputParser  # , StrOutputParser
from tqdm import tqdm

from src.constants.llm import (
    BATCH_SIZE,
    DO_SAMPLE,
    LLM_MODEL,
    MAX_NEW_TOKEN,
    RETURN_FULL_TEXT,
    TEMPERATURE,
)
from src.constants.paths import URL_DATASET_PREDICTED, URL_DATASET_PROMPTS, URL_LABELS
from src.constants.utils import DEVICE
from src.llm.build_llm import build_llm_model
from src.response.response_llm import LLMResponse, process_response
from src.utils.data import get_file_system, split_into_batches
from src.utils.mapping import lang_mapping

fs = get_file_system()

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
    device=DEVICE,
)


with fs.open(URL_LABELS) as f:
    labels = pd.read_csv(f, dtype=str)

parser = PydanticOutputParser(pydantic_object=LLMResponse)


for lang in lang_mapping.lang_iso_2[0]:
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

    data["lang"] = data["lang"].str.replace("lang=", "")

    batches = split_into_batches(data["prompt"].to_list(), BATCH_SIZE)

    responses = []
    for batch in tqdm(batches, total=len(batches)):
        response_batch = llm.generate(prompts=batch)
        responses.append(response_batch.generations)

    data.loc[:, "raw_responses"] = [response[0].text for response in sum(responses, [])]

    results = []
    for row in tqdm(data.itertuples(), total=data.shape[0]):
        result = process_response(row, parser, labels)
        results.append(result)

    pq.write_to_dataset(
        pa.Table.from_pylist(results),
        root_path=URL_DATASET_PREDICTED,
        partition_cols=["lang", "codable"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )
