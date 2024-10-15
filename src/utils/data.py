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
    )


def extract_info(input_str: str, paragraphs: list):
    """
    Extract the description and examples from the input string.
    """

    # Split the string by newlines
    lines = [line.strip() for line in input_str.split("\n")]

    # Store the first element of the list (the description)
    result = [lines[0].strip()]
    if len(paragraphs) == 1 and "description" in paragraphs:
        return "\n".join(result)

    if "tasks" in paragraphs:
        # Find the index of the examples
        try:
            # Try to find either of the example strings
            tasks_index = next(
                lines.index(task)
                for task in [
                    "Tasks include -",
                    "In such cases tasks would include -",
                    "In such cases tasks performed would include -",
                    "In such instances tasks would include -",
                ]
                if task in lines
            )
        except (ValueError, StopIteration):
            return "\n".join(result)

        # add examples
        result.append(lines[tasks_index])
        for line in lines[tasks_index + 1 :]:
            if line.startswith("("):
                result.append(line.strip())
            else:
                break

    if "examples" in paragraphs:
        # Find the index of the examples
        try:
            # Try to find either of the example strings
            examples_index = next(
                lines.index(example)
                for example in [
                    "Examples of the occupations classified here:",
                    "Example of the occupations classified here:",
                ]
                if example in lines
            )
        except (ValueError, StopIteration):
            return "\n".join(result)  # Return only the description

        # add examples
        result.append(lines[examples_index])
        for line in lines[examples_index + 1 :]:
            if line.startswith("-"):
                result.append(line.strip())
            else:
                break

    return "\n".join(result)


eol_regex = re.compile(r"\r|\n")
multispace_regex = re.compile(r"\s\s+")
html_regex = re.compile(r"<[^<]+?>")
white_regex = re.compile(r"\xa0")
star_regex = re.compile(r"(\*[\s]*)+")
url_regex = re.compile(r"(https?://(?:www\.)?[^\s/$.?#].[^\s]*)")
javascript_regex = re.compile(r"You need to enable JavaScript to run this app")
captcha_regex = re.compile(r"reCAPTCHA check page")


# Function to clean text using all relevant regex
def clean_text(text):
    """
    Clean text by removing HTML tags, URLs, and other irrelevant information.
    """
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    text = html_regex.sub(" ", text)
    text = url_regex.sub(" ", text)
    text = javascript_regex.sub(" ", text)
    text = captcha_regex.sub(" ", text)
    text = star_regex.sub(" ", text)
    text = white_regex.sub(" ", text)
    text = multispace_regex.sub(" ", text)
    return text.strip()
