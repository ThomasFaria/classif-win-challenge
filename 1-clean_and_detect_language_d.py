import re  # Regular expressions for text cleaning
import pandas as pd  # Data manipulation
import pyarrow as pa  # For efficient data representation
import pyarrow.parquet as pq  # For reading and writing Parquet files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting long texts into chunks

# Paths and utility imports
from src.constants.paths import URL_DATASET, URL_DATASET_WITH_LANG
from src.detect_lang.detect import detect_language, process_data_lang_detec  # Language detection functions
from src.utils.data import get_file_system  # File system handler
from src.utils.mapping import id_881693105_desc, lang_mapping  # Specific description for a job ID and language mappings

# Constants
DESC_CUTOFF_SIZE = 500  # Maximum number of characters for truncated descriptions

# Get the file system handler for interacting with storage
fs = get_file_system()

# Regular expression to match line endings (for cleaning)
eol_regex = re.compile(r"\r|\n")

# Load the dataset as a pandas DataFrame from a CSV file
with fs.open(URL_DATASET) as f:
    data = pd.read_csv(f, dtype=str)  # Read all columns as strings

# Manually set a specific job description for a particular ID
data.loc[data["id"] == "881693105", "description"] = id_881693105_desc

# Clean the data and extract job titles (or other necessary info) using a custom processing function
data = process_data_lang_detec(data)

# Detect the language and calculate a confidence score for each job description
data[["lang", "score"]] = (
    data["description_clean"]  # Cleaned job descriptions
    .str.replace(eol_regex, " ", regex=True)  # Replace line endings with spaces
    .apply(detect_language)  # Detect language and score
    .apply(pd.Series)  # Convert the result (tuple) into separate columns for language and score
)

# Mark descriptions with low confidence scores (<0.4) as "undefined" (lang = "un")
data["lang"] = data.apply(lambda row: "un" if row["score"] < 0.4 else row["lang"], axis=1)

# Further refine the language column: if the detected language is not in the EU language list, mark it as "un"
data["lang"] = data["lang"].where(data["lang"].isin(lang_mapping["lang_iso_2"]), "un")

# Initialize a text splitter to break job descriptions into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DESC_CUTOFF_SIZE,  # Maximum chunk size (in characters)
    chunk_overlap=0,  # No overlap between chunks
    length_function=len,  # Function to measure chunk length
    is_separator_regex=False,  # Don't treat separators as regular expressions
    separators=["\n\n", "\n", ".", "?", "!", ";", ":", ",", " ", ""],  # Separators used to split the text
)

# Iterate through the dataset and truncate each job description to the specified maximum length (DESC_CUTOFF_SIZE)
for idx, row in data.iterrows():
    splitted_text = text_splitter.split_text(row.description_clean)  # Split the description into chunks
    text_truncated = ""  # Initialize the truncated text
    i = 0
    # Concatenate chunks until the length exceeds the cutoff size or all chunks are used
    while (len(text_truncated) < DESC_CUTOFF_SIZE) and (i < len(splitted_text)):
        text_truncated += f" {splitted_text[i]}"  # Append the current chunk
        i += 1
    # Store the truncated text in a new column
    data.loc[idx, "description_truncated"] = text_truncated

# Save the updated dataset with the truncated descriptions to a Parquet file
pq.write_to_dataset(
    pa.Table.from_pandas(data),  # Convert the pandas DataFrame to a PyArrow Table
    root_path=URL_DATASET_WITH_LANG,  # Destination path for the Parquet file
    partition_cols=["lang"],  # Partition the dataset by language
    basename_template="part-{i}.parquet",  # Template for the output filenames
    existing_data_behavior="overwrite_or_ignore",  # Overwrite or ignore existing files
    filesystem=fs,  # Use the specified file system handler
)
