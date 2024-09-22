import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.constants.paths import URL_DATASET, URL_DATASET_WITH_LANG
from src.detect_lang.detect import detect_language, process_data_lang_detec
from src.utils.data import get_file_system
from src.utils.mapping import id_881693105_desc

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

pq.write_to_dataset(
    pa.Table.from_pandas(data),
    root_path=URL_DATASET_WITH_LANG,
    partition_cols=["lang"],
    basename_template="part-{i}.parquet",
    existing_data_behavior="overwrite_or_ignore",
    filesystem=fs,
)


# # Fonction pour extraire les 5 mots précédant l'abréviation
# def extract_job_title(line):
#     # Regex pour capturer les 5 mots précédant l'abréviation
#     match = re.search(r"(\b\w+\b[\s,]*){1,5}(?=\s*(" + abbreviations + "))", line, re.IGNORECASE)
#     if match:
#         return match.group().strip()
#     return None

# # Appliquer la fonction extract_job_title aux colonne 'description' et 'title' pour extraire les libellés de poste
# data["description_job_title"] = data["description"].apply(extract_job_title)
# data["title_job_title"] = data["title"].apply(extract_job_title)