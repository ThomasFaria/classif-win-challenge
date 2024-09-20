import html
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.constants.paths import URL_DATASET, URL_DATASET_WITH_LANG
from src.detect_lang.detect import detect_language
from src.utils.data import get_file_system
from src.utils.mapping import id_881693105_desc

eol_regex = re.compile(r"\r|\n")
multispace_regex = re.compile(r"\s\s+")
html_regex = re.compile(r"<[^<]+?>")
white_regex = re.compile(r"\xa0")
punctuation_regex = re.compile(r"[^\w\s]")
underscore_regex = re.compile(r"_")
star_regex = re.compile(r"(\*[\s]*)+")

# Liste des abréviations à détecter
abbreviations = r"\(?h/f\)?|\(?m/f\)?|\(?m/w\)?|\(?m/v\)?|\(?m/k\)?|\(?m/n\)?|\(?m/ž\)?|\(?f/n\)?|\(?b/f\)?|\(?άν/γυν\)?|\(?м/ж\)?"


# Fonction pour extraire les 5 mots précédant l'abréviation
def extract_job_title(line):
    # Regex pour capturer les 5 mots précédant l'abréviation
    match = re.search(r"(\b\w+\b[\s,]*){1,5}(?=\s*(" + abbreviations + "))", line, re.IGNORECASE)
    if match:
        return match.group().strip()
    return None


fs = get_file_system()

with fs.open(URL_DATASET) as f:
    data = pd.read_csv(f, dtype=str)

# Fill description when data is an image
data.loc[data["id"] == "881693105", "description"] = id_881693105_desc
# Remplacer les valeurs nulles ou vides dans 'description' par les valeurs de 'title'
data["description"] = data["description"].fillna(data["title"])
# Create description_clean from description
data["description_clean"] = data["description"][data["description"].notna()].apply(html.unescape)
data["description_clean"] = (
    data["description_clean"]
    .str.replace(eol_regex, " ", regex=True)
    .str.replace(html_regex, " ", regex=True)
    .str.replace(star_regex, " <ANONYMOUS> ", regex=True)
    .str.replace(white_regex, " ", regex=True)
    .str.replace(multispace_regex, " ", regex=True)
    .str.strip()
)

# Create title_clean from description
data["title_clean"] = data["title"][data["title"].notna()].apply(html.unescape)
data["title_clean"] = (
    data["title_clean"]
    .str.replace(eol_regex, " ", regex=True)
    .str.replace(html_regex, " ", regex=True)
    .str.replace(star_regex, " <ANONYMOUS> ", regex=True)
    .str.replace(white_regex, " ", regex=True)
    .str.replace(multispace_regex, " ", regex=True)
    .str.strip()
)

data[["lang", "score"]] = (
    data["description_clean"][data["description_clean"].notna()]
    .apply(detect_language)
    .apply(pd.Series)
)

# Appliquer la fonction extract_job_title aux colonne 'description' et 'title' pour extraire les libellés de poste
data["description_job_title"] = data["description"].apply(extract_job_title)
data["title_job_title"] = data["title"].apply(extract_job_title)

pq.write_to_dataset(
    pa.Table.from_pandas(data),
    root_path=URL_DATASET_WITH_LANG,
    partition_cols=["lang"],
    basename_template="part-{i}.parquet",
    existing_data_behavior="overwrite_or_ignore",
    filesystem=fs,
)
