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
        #       token=os.environ["AWS_SESSION_TOKEN"],
    )


def split_into_batches(list, batch_size):
    # Create sublists of size `batch_size` from `list`
    return [list[i : i + batch_size] for i in range(0, len(list), batch_size)]


def truncate_txt(input_str: str, phrases: list):
    """
    Truncate the input string before the first occurrence of any of the given phrases.

    Parameters:
    input_str (str): The input text.
    phrases (list): A list of phrases after which the text should be truncated.

    Returns:
    str: The truncated string if any phrase is found, otherwise the original string.
    """
    # Initialize position to a value greater than the length of the input string
    first_phrase_pos = len(input_str)

    # Find the earliest occurrence of any phrase
    for phrase in phrases:
        phrase_pos = input_str.find(phrase)

        # If the phrase is found and occurs earlier than the current found position
        if phrase_pos != -1 and phrase_pos < first_phrase_pos:
            first_phrase_pos = phrase_pos

    # If no phrase is found, return the original string
    if first_phrase_pos == len(input_str):
        return input_str

    # Return the truncated string up to the first found phrase
    return input_str[:first_phrase_pos]
