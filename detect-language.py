from src.utils.data import get_file_system
import pandas as pd
from transformers import pipeline
import pyarrow.parquet as pq
import pyarrow as pa
import torch

MODEL_NAME = "papluca/xlm-roberta-base-language-detection"
URL_OUT = "s3://projet-dedup-oja/challenge_classification/raw-data/wi_dataset.parquet"
device = "cuda" if torch.cuda.is_available() else "cpu"

fs = get_file_system()

with fs.open("s3://projet-dedup-oja/challenge_classification/raw-data/wi_dataset.csv") as f:
    data = pd.read_csv(f, dtype=str)

# 2 lines with no description
data.fillna("", inplace=True)

pipe = pipeline("text-classification", model=MODEL_NAME, device=device)

text = data.loc[:, "description"].to_list()

results = pipe(text, top_k=1, truncation=True)
df = data.merge(pd.DataFrame([d[0] for d in results]), left_index=True, right_index=True)
df.set_index("id")

pq.write_table(
    pa.Table.from_pandas(df),
    URL_OUT,
    filesystem=fs,
)
