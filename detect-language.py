from src.utils.data import get_file_system
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from ftlangdetect import detect
import re
import html
            
# Detect text language
def detect_language(text):
    '''
    Detect text language 
    return lang and score
    '''
    result = detect(text)
    return result['lang'], result['score']


eol_regex = re.compile(r'\r|\n')
multispace_regex = re.compile(r'\s\s+')
html_regex = re.compile(r'<[^<]+?>')
white_regex = re.compile(r'\xa0')
punctuation_regex = re.compile(r'[^\w\s]')
underscore_regex = re.compile(r'_')

fs = get_file_system()

with fs.open("s3://projet-dedup-oja/challenge_classification/raw-data/wi_dataset.csv") as f:
    data = pd.read_csv(f, dtype=str)

# 2 lines with no description
data.fillna("", inplace=True)

data['description'] = data['description'][data['description'].notna()].apply(html.unescape)
data['description'] = data['description'].str.lower().str.replace(eol_regex,' ',regex=True).str.replace(html_regex,' ',regex=True)\
                            .str.replace(white_regex,' ',regex=True).str.replace(multispace_regex,' ',regex=True)\
                            .str.strip()

data[['lang', 'score']] = data['description'][data['description'].notna()].apply(detect_language).apply(pd.Series)

data.set_index("id")  # TODO

pq.write_to_dataset(
    pa.Table.from_pandas(data),
    root_path=URL_OUT,
    partition_cols=["lang"],
    basename_template="part-{i}.parquet",
    existing_data_behavior="overwrite_or_ignore",
    filesystem=fs,
)
