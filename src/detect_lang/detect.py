from ftlangdetect import detect


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
        raise ValueError(f"Language detection failed: {e}")
