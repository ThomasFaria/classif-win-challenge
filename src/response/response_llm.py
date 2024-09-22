from typing import Optional
import json

from pydantic import BaseModel, Field

from src.constants.utils import ISCO_CODES


class LLMResponse(BaseModel):
    """Represents a response model for classification code assignment.

    Attributes:
        codable (bool): True if enough information is provided to decide
            classification code.
        class_code (Optional[str]): ISCO classification code Empty if codable=False
        likelihood (Optional[float]): Likelihood of this soc_code with a value between 0 and 1.
    """

    codable: bool = Field(
        description="""True if enough information is provided to decide
        classification code, False otherwise."""
    )

    class_code: Optional[str] = Field(
        description="""ISCO classification code Empty if codable=False.""",
        default=None,
    )

    likelihood: Optional[float] = Field(
        description="Likelihood of this class_code with value between 0 and 1."
    )


class TranslatorResponse(BaseModel):
    """Represents a response model for translation assignment.

    Attributes:
        translated (bool): True if translated.
        translation (Optional[str]): Translation
    """

    translated: bool = Field()

    translation: Optional[str] = Field()


def process_response(row: tuple, parser, labels) -> dict:
    """
    Processes a row of raw responses by parsing and validating the response,
    and then returns a dictionary with additional information.

    Args:
        row (tuple): A tuple representing a row of data.
                         Should contain 'raw_responses', 'id', 'lang', and 'prompt'.
        parser (Parser): A parser object with a `parse` method used to validate and parse the raw response.
        labels (pd.DataFrame): A DataFrame containing the label information.
                               Must have columns 'code' and 'label'.

    Returns:
        dict: A dictionary containing validated and parsed response data along with additional metadata:
              - 'id': The ID of the row.
              - 'label_code': The label corresponding to the parsed class code, if valid; None otherwise.
              - 'lang': The language of the row.
              - 'prompt': The prompt from the row.
              - other parsed response details from `validated_response.dict()`.

    Raises:
        ValueError: If parsing fails or the class code is invalid.
    """
    response = row.raw_responses  # Extract the raw response data from the row.
    row_id = row.id  # Extract the row's unique identifier.

    try:
        # Attempt to parse the response using the provided parser.
        validated_response = parser.parse(response)
    except ValueError as parse_error:
        # Log an error and return an un-codable response if parsing fails.
        print(f"Error processing row with id {row_id}: {parse_error}")
        validated_response = LLMResponse(codable=False, class_code=None, likelihood=None)

    # Validate the parsed class code against ISCO_CODES (International Standard Classification of Occupations).
    if validated_response.class_code not in ISCO_CODES:
        # Log an error if the class code is invalid.
        print(
            f"Error processing row with id {row_id}: Code not in the ISCO list --> {validated_response.class_code}"
        )
        # Mark the response as un-codable and clear the invalid class code and likelihood.
        validated_response.codable = False
        validated_response.likelihood = None
        label_code = None
    else:
        # If the class code is valid, map it to the corresponding label.
        label_code = labels.loc[labels["code"] == validated_response.class_code, "label"].values[0]

    # Return a dictionary containing the processed response details along with row metadata.
    return {
        **validated_response.dict(),  # Unpack the parsed response details.
        "id": row_id,  # Include the row's ID.
        "label_code": label_code,  # Add the label corresponding to the class code, if valid.
        "lang": row.lang,  # Include the language metadata.
        "prompt": row.prompt,  # Include the original prompt metadata.
    }


def process_translation(row: tuple, translated_col: str, parser) -> dict:
    col = getattr(row, translated_col)
    row_id = row.id  # Extract the row's unique identifier.

    try:
        # Attempt to parse the response using the provided parser.
        validated_response = parser.parse(col)
        # Ensure translation is a string, not a dict. Sometimes it returns a dict but it contains the translation
        if isinstance(validated_response.translation, dict):
            validated_response.translation = json.dumps(validated_response.translation)

    except ValueError as parse_error:
        # Log an error and return an un-codable response if parsing fails.
        print(f"Error processing row with id {row_id}: {parse_error}")
        validated_response = TranslatorResponse(translated=False, translation=None)

    # Return a dictionary containing the processed response details along with row metadata.
    return {
        **validated_response.dict(),  # Unpack the parsed response details.
        "id": row_id,  # Include the row's ID.
    }
