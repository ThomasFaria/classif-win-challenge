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
    PROMPT_MAX_TOKEN,
    RETURN_FULL_TEXT,
    TEMPERATURE,
)
from src.constants.paths import (
    CHROMA_DB_LOCAL_DIRECTORY,
    URL_DATASET_PREDICTED,
    URL_DATASET_TRANSLATED,
    URL_LABELS,
)
from src.constants.utils import DEVICE, ISCO_CODES
from src.constants.vector_db import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_CODE_RETRIEVED,
    SEARCH_ALGO,
)
from src.llm.build_llm import build_llm_model
from src.prompting.prompts import CLASSIF_PROMPT_TEMPLATE, format_docs, generate_valid_prompt
from src.response.response_llm import LLMResponse
from src.utils.data import get_file_system
from src.utils.mapping import lang_mapping
from src.vector_db.loaders import load_retriever

TITLE_COLUMN = "title"
DESCRIPTION_COLUMN = "translation"

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
)


retriever = load_retriever(
    embedding_model=EMBEDDING_MODEL,
    collection_name=COLLECTION_NAME,
    db_directory=CHROMA_DB_LOCAL_DIRECTORY,
    search_algo=SEARCH_ALGO,
    device=DEVICE,
    max_code_retrieved=MAX_CODE_RETRIEVED,
)

with fs.open(URL_LABELS) as f:
    labels = pd.read_csv(f, dtype=str)

parser = PydanticOutputParser(pydantic_object=LLMResponse)


def process_row(row, parser):
    """Process a single row and return the prediction."""
    description = row[TITLE_COLUMN]
    title = row[DESCRIPTION_COLUMN]
    id = row.id

    try:
        # Retrieve documents and make sure a document is not retrieved twice
        retrieved_docs = retriever.invoke(" ".join([title, description]))
        retrieved_docs_unique = []
        for item in retrieved_docs:
            if item not in retrieved_docs_unique:
                retrieved_docs_unique.append(item)

        # Generate the prompt and include the number of documents
        prompt_template = PromptTemplate.from_template(
            template=CLASSIF_PROMPT_TEMPLATE,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        prompt, num_documents_included = generate_valid_prompt(
            prompt_template,
            PROMPT_MAX_TOKEN,
            tokenizer,
            title=title,
            description=description,
            retrieved_docs=retrieved_docs_unique,
        )

        # Chain and response generation
        chain = prompt_template | llm | parser
        response = chain.invoke(
            {
                "title": title,
                "description": description,
                "proposed_categories": format_docs(retrieved_docs_unique[:num_documents_included]),
            }
        )

    except ValueError as parse_error:
        print(f"Error processing row with id {id}: {parse_error}")
        response = LLMResponse(codable=False)

    # Validate class code
    if response.class_code not in ISCO_CODES:
        response.codable = False
        response.class_code = None
        response.likelihood = None
        label_code = None
        print(
            f"Error processing row with id {id}: Code proposed is not in the ISCO list --> {response.class_code}"
        )
    else:
        label_code = labels.loc[labels["code"] == response.class_code, "label"].values[0]

    return response.dict() | {"id": id, "label_code": label_code}


for lang in lang_mapping.lang_iso_2:
    print(f"Processing for language: {lang}")

    # Load the dataset
    data = (
        ds.dataset(
            URL_DATASET_TRANSLATED.replace("s3://", ""),
            partitioning=["lang"],
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .filter((ds.field("lang") == f"lang={lang}"))
        .to_pandas()
    ).head(10)

    if data.empty:
        print(f"No data found for language {lang}. Skipping...")
        continue

    data["lang"] = data["lang"].str.replace("lang=", "")

    results = []
    for row in tqdm(data.itertuples(), total=data.shape[0]):
        result = process_row(row, parser)
        results.append(result)

    # Merge results back to the original dataset
    predictions = data.merge(pd.DataFrame(results), on="id")
    pq.write_to_dataset(
        pa.Table.from_pandas(predictions),
        root_path=URL_DATASET_PREDICTED,
        partition_cols=["lang"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )
