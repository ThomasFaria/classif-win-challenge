import json
from typing import Optional

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.constants.utils import ISCO_CODES


class LLMResponse(BaseModel):
    """Represents a response model for classification code assignment."""

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
    """Represents a response model for extraction and translation assignment."""

    job_desc_extracted: bool = Field(description="True if the job description has been extracted.")
    title: Optional[str] = Field(description="The job title in english.")
    description: Optional[str] = Field(description="The job description in english.")
    keywords: Optional[list[str]] = Field(
        description="A list of important terms focused on the job-specific skills, responsibilities, fields, or qualifications, and should not include geographic locations or overly broad categories. Limit to a maximum of 5 terms."
    )


def process_response(row: tuple, parser: PydanticOutputParser, labels: pd.DataFrame) -> dict:
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
        # Ensure class_code is a string, not a dict. Sometimes it returns a dict but it contains the class_code
        if isinstance(validated_response.class_code, dict):
            validated_response.class_code = json.dumps(validated_response.class_code)

    except ValueError as parse_error:
        # Log an error and return an un-codable response if parsing fails.
        print(f"Error processing row with id {row_id}: {parse_error}")
        validated_response = LLMResponse(
            codable=False,
            class_code=None,
            likelihood=None,  # TODO: fix itvalidated_response.translation
        )

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
    }


def process_translation(row: tuple, parser: PydanticOutputParser) -> dict:
    """
    Processes a row of raw translations by parsing and validating the response,
    and then returns a dictionary with additional information.

    Args:
        row (tuple): A tuple representing a row of data.
                        Should contain 'raw_translations' and 'id'.
        parser (Parser): A parser object with a `parse` method used to validate and parse the raw response.

    Returns:
        dict: A dictionary containing validated and parsed response data along with additional metadata:
              - 'id': The ID of the row.
              - other parsed response details from `validated_response.dict()`.
    """

    response = row.raw_translations  # Extract the raw response data from the row.
    row_id = row.id  # Extract the row's unique identifier.

    try:
        # Attempt to parse the response using the provided parser.
        validated_response = parser.parse(response)
        # Ensure title is a string, not a dict. Sometimes it returns a dict but it contains the title
        if isinstance(validated_response.title, dict):
            validated_response.title = json.dumps(validated_response.title)
        if isinstance(validated_response.description, dict):
            validated_response.description = json.dumps(validated_response.description)

    except ValueError as parse_error:
        # Log an error and return an un-codable response if parsing fails.
        print(f"Error processing row with id {row_id}: {parse_error}")
        validated_response = TranslatorResponse(
            job_desc_extracted=False, title=None, description=None, keywords=None
        )

    # Return a dictionary containing the processed response details along with row metadata.
    return {
        **validated_response.dict(),  # Unpack the parsed response details.
        "id": row_id,  # Include the row's ID.
    }
