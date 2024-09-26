import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.constants.paths import URL_LABELS, URL_LABELS_WITH_LANG_R, URL_LABELS_WITH_LANG_W
from src.utils.data import get_file_system
from src.utils.mapping import lang_mapping

fs = get_file_system()

list_label = []

# Liste des langues à exclure
exclusions = ["en", "un"]
# Filtrer la liste des langues en excluant 'en' et 'un'
lang_list = [lang for lang in lang_mapping["lang_iso_2"] if lang not in exclusions]

for lang in lang_list:
    with fs.open(f"{URL_LABELS_WITH_LANG_R}/ISCOGroups_{lang}.csv") as f:
        isco = pd.read_csv(f, dtype=str)
        isco = (
            isco[isco["code"].str.len() == 4]
            .loc[:, ["code", "preferredLabel", "conceptUri"]]
            .reset_index(drop=True)
            .rename(
                columns={"iscoGroup": "code", "preferredLabel": "label", "conceptUri": "isco_uri"}
            )
        )
        isco.loc[:, "lang"] = lang

    with fs.open(f"{URL_LABELS_WITH_LANG_R}/occupations_{lang}.csv") as f:
        occupations = pd.read_csv(f, dtype=str)
        occupations = (
            occupations.groupby("iscoGroup")
            .agg(
                {
                    "description": " ".join,  # Concatenate descriptions with a space
                }
            )
            .reset_index()
            .rename(columns={"iscoGroup": "code"})
        )
        list_label.append(
            pd.merge(isco, occupations, on="code").loc[
                :, ["code", "label", "description", "isco_uri", "lang"]
            ]
        )

with fs.open(URL_LABELS) as f:
    labels_en = pd.read_csv(f, dtype=str)
    labels_en.loc[:, "lang"] = "en"

list_label.append(labels_en)
final_df = pd.concat(list_label, ignore_index=True)
pq.write_to_dataset(
    pa.Table.from_pandas(final_df),
    root_path=URL_LABELS_WITH_LANG_W,
    partition_cols=["lang"],
    basename_template="part-{i}.parquet",
    existing_data_behavior="overwrite_or_ignore",
    filesystem=fs,
)
