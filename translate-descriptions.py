from src.utils.data import get_file_system
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pyarrow.parquet as pq
import pyarrow as pa
from src.utils.mapping import lang_mapping
from src.utils.data import split_into_batches
import pyarrow.dataset as ds
from tqdm import tqdm
from src.translation.translate import translate_batch
from src.config.constants import (
    URL_DATASET_WITH_LANG,
    URL_DATASET_TRANSLATED,
    DEVICE,
    TRANSLATOR_MODEL,
    BATCH_SIZE,
    MAX_LENGTH_TO_TRANSLATE,
    MAX_LENGTH_TRANSLATED,
)

fs = get_file_system()

model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)


for lang_iso_2, lang_iso_3 in zip(lang_mapping.lang_iso_2, lang_mapping.lang_iso_3):
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
        data.loc[:, "translation"] = data["title"] + " " + data["description"]
    else:
        print(f"Translating texts from {lang_iso_3} to English")
        txt_to_translate = (data["title"] + " " + data["description"]).to_list()
        batches = split_into_batches(txt_to_translate, BATCH_SIZE)

        translations = []
        for batch in tqdm(batches, total=len(batches)):
            translated_texts = translate_batch(
                batch,
                lang_iso_3,
                tokenizer,
                model,
                DEVICE,
                max_length_encoded=MAX_LENGTH_TO_TRANSLATE,
                max_length_decoded=MAX_LENGTH_TRANSLATED,
            )
            translations.append(translated_texts)

        data.loc[:, "translation"] = sum(translations, [])  # flatten list of lists

    pq.write_to_dataset(
        pa.Table.from_pandas(data),
        root_path=URL_DATASET_TRANSLATED,
        partition_cols=["lang"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )
