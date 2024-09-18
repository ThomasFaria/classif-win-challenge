from src.utils.data import get_file_system
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pyarrow.parquet as pq
import pyarrow as pa
from src.utils.mapping import lang_mapping
import pyarrow.dataset as ds

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
URL_IN = "projet-dedup-oja/challenge_classification/processed-data/wi_dataset_by_lang"
URL_OUT = (
    "s3://projet-dedup-oja/challenge_classification/processed-data/wi_dataset_by_lang_translated"
)
MAX_LENGTH_ENCODED = 512
MAX_LENGTH_DECODED = 512

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
        txt_to_translate = (data["title"] + " " + data["description"]).to_list()
        print(f"Translating texts from {lang_iso_3} to English")
        tokenizer.src_lang = lang_iso_3
        encoded_txt = tokenizer(
            txt_to_translate,
            return_tensors="pt",
            padding=True,
            truncation=True,  # We assume the most important part of the text is at the beggining
            max_length=MAX_LENGTH_ENCODED,
        ).to(device)
        print(f"The shape of encoded txt is: {encoded_txt["input_ids"].shape}")

        generated_tokens = model.generate(
            **encoded_txt,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=MAX_LENGTH_DECODED,  # NLLB-200  was trained with input lengths not exceeding 512 tokens
        )
        print(f"The shape of decoded txt is: {generated_tokens.shape}")

        results = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        data.loc[:, "translation"] = results

    data.set_index("id")  # TODO

    pq.write_to_dataset(
        pa.Table.from_pandas(data),
        root_path=URL_OUT,
        partition_cols=["label"],
        basename_template="part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        filesystem=fs,
    )
