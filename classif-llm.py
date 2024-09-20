from src.utils.data import get_file_system
from src.llm.build_llm import build_llm_model
from src.vector_db.loaders import load_retriever
from src.utils.mapping import lang_mapping

import os
from src.prompting.prompts import RAG_PROMPT_TEMPLATE, format_docs
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser  # , StrOutputParser
import pyarrow.dataset as ds

from tqdm import tqdm
from src.response.response_llm import LLMResponse
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


from src.constants.utils import DEVICE, ISCO_CODES
from src.constants.llm import MAX_NEW_TOKEN, RETURN_FULL_TEXT, DO_SAMPLE, TEMPERATURE, LLM_MODEL
from src.constants.vector_db import (
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    SEARCH_ALGO,
    MAX_CODE_RETRIEVED,
)
from src.constants.paths import (
    CHROMA_DB_LOCAL_DIRECTORY,
    URL_LABELS,
    URL_DATASET_TRANSLATED,
    URL_DATASET_PREDICTED,
)


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


for lang in lang_mapping.lang_iso_2:
    print(f"Prediction for language: {lang}")
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
    )

    if data.empty:
        continue

    # Reformat partionnning column
    data.loc[:, "lang"] = data.loc[:, "lang"].str.replace("lang=", "")

    results = []

    for row in tqdm(data.itertuples(), total=data.shape[0]):
        description = row.translation
        title = row.title
        id = row.id

        # Ici faut voir ce qu'on fait mais faut virer/nettoyer/selectionner les documents
        retrieved_docs = retriever.invoke(" ".join([title, description]))
        retrieved_docs_unique = []
        for item in retrieved_docs:
            if item not in retrieved_docs_unique:
                retrieved_docs_unique.append(item)

        parser = PydanticOutputParser(pydantic_object=LLMResponse)
        prompt_template = PromptTemplate.from_template(
            template=RAG_PROMPT_TEMPLATE,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # TODO: make sure the prompt is not to long
        # prompt = prompt_template.format(
        #     **{
        #         "title": title,
        #         "description": description,
        #         "proposed_categories": format_docs(retrieved_docs_unique),
        #     }
        # )

        # On crée la chaine de traitement
        chain = prompt_template | llm | parser

        # On interroge le LLM
        try:
            response = chain.invoke(
                {
                    "title": title,
                    "description": description,
                    "proposed_categories": format_docs(retrieved_docs_unique),
                },
            )
        except ValueError as parse_error:
            print(f"Unable to parse llm response: {str(parse_error)}")
            # logger.error(f"Unable to parse llm response: {str(parse_error)}")

            response = LLMResponse(
                codable=False,
            )

        # Make sure the code is part of the list of ISCO codes
        if response.class_code not in ISCO_CODES:
            response.codable = False
            response.class_code = None
            response.likelihood = None
            label_code = None
        else:
            label_code = labels.loc[labels["code"] == response.class_code, "label"].values[0]

        results.append(response.dict() | {"id": id, "label_code": label_code})

    predictions = data.merge(pd.DataFrame(results), on="id")

    pq.write_to_dataset(
        pa.Table.from_pandas(predictions),
        root_path=URL_DATASET_PREDICTED,
        partition_cols=["lang"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )