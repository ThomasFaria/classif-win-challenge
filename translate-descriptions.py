from src.utils.data import get_file_system
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pyarrow.parquet as pq
import pyarrow as pa
from src.utils.mapping import lang_mapping
from src.utils.data import split_into_batches
import pyarrow.dataset as ds
from tqdm import tqdm
from src.translation.translate import translate_batch

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
URL_IN = "projet-dedup-oja/challenge_classification/processed-data/wi_dataset_by_lang"
URL_OUT = (
    "s3://projet-dedup-oja/challenge_classification/processed-data/wi_dataset_by_lang_translated"
)
MAX_LENGTH_ENCODED = 512
MAX_LENGTH_DECODED = 512
BATCH_SIZE = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

fs = get_file_system()

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


for lang_iso_2, lang_iso_3 in zip(lang_mapping.lang_iso_2, lang_mapping.lang_iso_3):
    data = (
        ds.dataset(
            URL_IN,
            partitioning=["label"],
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .filter((ds.field("label") == f"label={lang_iso_2}"))
        .to_pandas()
    )

    if data.empty:
        continue

    # Reformat partionnning column
    data.loc[:, "label"] = data.loc[:, "label"].str.replace("label=", "")

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
                device,
                max_length_encoded=MAX_LENGTH_ENCODED,
                max_length_decoded=MAX_LENGTH_DECODED,
            )
            translations.append(translated_texts)

        data.loc[:, "translation"] = sum(translations, [])  # flatten list of lists

    data.set_index("id")  # TODO

    pq.write_to_dataset(
        pa.Table.from_pandas(data),
        root_path=URL_OUT,
        partition_cols=["label"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )
