from typing import Optional

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Represents a response model for classification code assignment.

    Attributes:
        codable (bool): True if enough information is provided to decide
            classification code, False otherwise.
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
