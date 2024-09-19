from src.utils.data import get_file_system
from src.detect_lang.detect import detect_language
from src.utils.mapping import id_881693105_desc

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

import re
import html

URL_OUT = "s3://projet-dedup-oja/challenge_classification/processed-data/wi_dataset_by_lang"

eol_regex = re.compile(r"\r|\n")
multispace_regex = re.compile(r"\s\s+")
html_regex = re.compile(r"<[^<]+?>")
white_regex = re.compile(r"\xa0")
punctuation_regex = re.compile(r"[^\w\s]")
underscore_regex = re.compile(r"_")

fs = get_file_system()

with fs.open("s3://projet-dedup-oja/challenge_classification/raw-data/wi_dataset.csv") as f:
    data = pd.read_csv(f, dtype=str)

data.loc[data["id"] == "881693105", "description"] = id_881693105_desc
data["description_clean"] = data["description"][data["description"].notna()].apply(html.unescape)
data["description_clean"] = (
    data["description_clean"]
    .str.lower()
    .str.replace(eol_regex, " ", regex=True)
    .str.replace(html_regex, " ", regex=True)
    .str.replace(white_regex, " ", regex=True)
    .str.replace(multispace_regex, " ", regex=True)
    .str.strip()
)

data[["lang", "score"]] = (
    data["description_clean"][data["description_clean"].notna()]
    .apply(detect_language)
    .apply(pd.Series)
)
data.set_index("id")  # TODO

pq.write_to_dataset(
    pa.Table.from_pandas(data),
    root_path=URL_OUT,
    partition_cols=["lang"],
    basename_template="part-{i}.parquet",
    existing_data_behavior="overwrite_or_ignore",
    filesystem=fs,
)
