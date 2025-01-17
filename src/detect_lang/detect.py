import pandas as pd
from ftlangdetect import detect

from src.utils.data import clean_text


def detect_language(text: str) -> tuple:
    """
    Detect the language of a given text and return both the language code and confidence score.

    Parameters:
    ----------
    text : str
        The input text whose language needs to be detected.

    Returns:
    -------
    tuple
        A tuple containing:
        - lang (str): The detected language code (ISO 639-1 format).
        - score (float): The confidence score of the detected language (range between 0 and 1).

    Raises:
    ------
    ValueError:
        If the input text is empty or if the language detection fails.

    Example:
    -------
    >>> detect_language("This is a test sentence.")
    ('en', 0.99)
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty.")

    try:
        # Detect the language using langid's classify function
        result = detect(text)
        return result["lang"], result["score"]
    except Exception as e:
        print(text)
        raise ValueError(f"Language detection failed: {e}")


def process_data_lang_detec(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input data by detecting the language of the 'description' and 'title' columns.

    Parameters:
    ----------
    data : pd.DataFrame
        The input data containing 'description' and 'title' columns.

    Returns:
    -------
    pd.DataFrame
        The processed data with additional columns 'description_lang' and 'title_lang' containing the detected language codes.
    """

    # Apply the text cleaning function to both 'description' and 'title'
    data["description_clean"] = data["description"].apply(clean_text)
    data["title_clean"] = data["title"].apply(clean_text)

    # Fill description where missing with title
    data["description"] = data["description"].fillna(data["title"])
    data["description_clean"] = data["description_clean"].fillna(data["title_clean"])
    data.loc[data["description_clean"].str.strip() == "", "description_clean"] = data["title_clean"]
    return data
