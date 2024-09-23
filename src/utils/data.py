import html
import os
import re

import s3fs


def get_file_system() -> s3fs.S3FileSystem:
    """
    Return the s3 file system.
    """
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        # token=os.environ["AWS_SESSION_TOKEN"],
    )


def split_into_batches(list, batch_size):
    # Create sublists of size `batch_size` from `list`
    return [list[i : i + batch_size] for i in range(0, len(list), batch_size)]


def extract_info(input_str):
    # Split the string by newlines
    lines = [line.strip() for line in input_str.split("\n")]

    # Store the first element of the list (the description)
    result = [lines[0].strip()]

    # Find the index of the examples
    try:
        examples_index = lines.index("Examples of the occupations classified here:")
    except ValueError:
        return result  # Return only the description

    # add examples
    result.append(lines[examples_index])
    for line in lines[examples_index + 1 :]:
        if line.startswith("-"):
            result.append(line.strip())
        else:
            break

    return "\n".join(result)


multispace_regex = re.compile(r"\s\s+")
html_regex = re.compile(r"<[^<]+?>")
white_regex = re.compile(r"\xa0")
punctuation_regex = re.compile(r"[^\w\s]")
underscore_regex = re.compile(r"_")
star_regex = re.compile(r"(\*[\s]*)+")

# Liste des abréviations à détecter
abbreviations = r"\(?h/f\)?|\(?m/f\)?|\(?m/w\)?|\(?m/v\)?|\(?m/k\)?|\(?m/n\)?|\(?m/ž\)?|\(?f/n\)?|\(?b/f\)?|\(?άν/γυν\)?|\(?м/ж\)?"


# Function to clean text using all relevant regex
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    text = html_regex.sub(" ", text)
    text = star_regex.sub(" <ANONYMOUS> ", text)
    text = white_regex.sub(" ", text)
    text = multispace_regex.sub(" ", text)
    return text.strip()
