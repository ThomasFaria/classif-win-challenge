import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.constants.paths import URL_DATASET, URL_DATASET_WITH_LANG
from src.detect_lang.detect import detect_language, process_data_lang_detec
from src.utils.data import get_file_system
from src.utils.mapping import id_881693105_desc, lang_mapping

DESC_CUTOFF_SIZE = 500

fs = get_file_system()
eol_regex = re.compile(r"\r|\n")

with fs.open(URL_DATASET) as f:
    data = pd.read_csv(f, dtype=str)

# Manually set specific description for one ID
data.loc[data["id"] == "881693105", "description"] = id_881693105_desc

# Process the dataset with cleaning and job title extraction
data = process_data_lang_detec(data)

# Detect language and score
data[["lang", "score"]] = (
    data["description_clean"]
    .str.replace(eol_regex, " ", regex=True)
    .apply(detect_language)
    .apply(pd.Series)
)
# Set lang to undefined when score are low
data["lang"] = data.apply(lambda row: "un" if row["score"] < 0.4 else row["lang"], axis=1)
# Set lang to undefined (lang="un") if not an EU language
data["lang"] = data["lang"].where(data["lang"].isin(lang_mapping["lang_iso_2"]), "un")

# Truncate the description
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DESC_CUTOFF_SIZE,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ".", "?", "!", ";", ":", ",", " ", ""],
)

for idx, row in data.iterrows():
    splitted_text = text_splitter.split_text(row.description_clean)
    text_truncated = ""
    i = 0
    while (len(text_truncated) < DESC_CUTOFF_SIZE) and (i < len(splitted_text)):
        text_truncated += f" {splitted_text[i]}"
        i += 1
    data.loc[idx, "description_truncated"] = text_truncated

# save to parquet file
pq.write_to_dataset(
    pa.Table.from_pandas(data),
    root_path=URL_DATASET_WITH_LANG,
    partition_cols=["lang"],
    basename_template="part-{i}.parquet",
    existing_data_behavior="overwrite_or_ignore",
    filesystem=fs,
)


