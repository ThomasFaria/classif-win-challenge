import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.constants.paths import URL_LABELS_WITH_LANG_R, URL_LABELS_WITH_LANG_W
from src.utils.mapping import lang_mapping

import s3fs
import os


def get_file_system() -> s3fs.S3FileSystem:
    """
    Return the s3 file system.
    """
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"],
    )


fs = get_file_system()

list_label = []

# Liste des langues à exclure
exclusions = ['en', 'un']
# Filtrer la liste des langues en excluant 'en' et 'un'
lang_list = [lang for lang in lang_mapping["lang_iso_2"] if lang not in exclusions]

for lang in lang_list:
    with fs.open(f"{URL_LABELS_WITH_LANG_R}/ISCOGroups_{lang}.csv") as f:
        isco = pd.read_csv(f, dtype=str)
        isco = isco[isco["code"].str.len() == 4].loc[:, ["code", "preferredLabel", "conceptUri"]].reset_index(drop=True).rename(columns={'iscoGroup': 'code', "preferredLabel" : "label", "conceptUri": "isco_uri"})
        isco.loc[:, "lang"] = lang

    with fs.open(f"{URL_LABELS_WITH_LANG_R}/occupations_{lang}.csv") as f:
        occupations = pd.read_csv(f, dtype=str)
        occupations = occupations.groupby('iscoGroup').agg({
            'description': ' '.join,  # Concatenate descriptions with a space
        }).reset_index().rename(columns={'iscoGroup': 'code'})
        list_label.append(pd.merge(isco, occupations, on="code").loc[:, ["code", "label", "description", "isco_uri", "lang"]])

final_df = pd.concat(list_label, ignore_index=True)
pq.write_to_dataset(
    pa.Table.from_pandas(final_df),
    root_path=URL_LABELS_WITH_LANG_W,
    partition_cols=["lang"],
    basename_template="part-{i}.parquet",
    existing_data_behavior="overwrite_or_ignore",
    filesystem=fs,
)
