import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config.constants import (
    BATCH_SIZE,
    DEVICE,
    MAX_LENGTH_TO_TRANSLATE,
    MAX_LENGTH_TRANSLATED,
    TRANSLATOR_MODEL,
    URL_DATASET_TRANSLATED,
    URL_DATASET_WITH_LANG,
)
from src.translation.translate import translate_batch
from src.utils.data import get_file_system, split_into_batches
from src.utils.mapping import lang_mapping

TITLE_COLUMN = "title_clean"
DESCRIPTION_COLUMN = "description_truncated"

fs = get_file_system()

model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_MODEL).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)

# Trunquer les descriptions avec recursiveTextsplitter -> + rajouter la phrase comprenant les mots du titre


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
        data.loc[:, "title_en"] = data[TITLE_COLUMN]
        data.loc[:, "description_en"] = data[DESCRIPTION_COLUMN]
    else:
        print(f"Translating texts from {lang_iso_3} to English")
        for col in [TITLE_COLUMN, DESCRIPTION_COLUMN]:
            txt_to_translate = data[col].to_list()
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

            data.loc[:, f"{col}_en"] = sum(translations, [])  # flatten list of lists

    pq.write_to_dataset(
        pa.Table.from_pandas(data),
        root_path=URL_DATASET_TRANSLATED,
        partition_cols=["lang"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )
