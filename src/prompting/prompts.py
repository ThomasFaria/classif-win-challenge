import pandas as pd

from langchain_community.document_loaders import DataFrameLoader
from langchain_core.prompts import PromptTemplate

from src.utils.mapping import lang_mapping

CLASSIF_PROMPT_TEMPLATE = """Your goal is to select the single most appropriate occupational category for the job based on the description, its title and a list of the most relevant occupational categories selected.
The job title and description are in their original language : {language}.

Job Ad Title:
{title}

Job Ad Description:
{description}

Relevant Occupational Categories:
{proposed_categories}

{format_instructions}
"""

TRANSLATION_PROMPT = """You will be provided with a possibly noisy job offer text in {language}. Your task is to extract and translate into English only the relevant information about the job title and the job description. Do not translate the entire text, focus solely on the content related to the job description, which includes the key responsibilities, tasks, and role details, so that the information can later be used to classify the job offer. Additionally, ensure that the job title is translated exactly as it appears.
Below are the job offer details:
-	Job Ad Title (to translate exactly in english): {title}
-	Job Ad Description (to extract and translate in enlish): {description}

{format_instructions}
"""

EXTRACTION_PROMPT = """You will be provided with a possibly noisy job offer. Your task is to extract only the relevant information about the job title and the job description. Focus solely on the content related to the job description, which includes the key responsibilities, tasks, and role details, so that the information can later be used to classify the job offer.
Below are the job offer details:
-	Job Ad Title: {title}
-	Job Ad Description: {description}

{format_instructions}
"""


def format_docs(docs: list):
    return "\n\n".join(
        [f"{doc.metadata['code']}: {doc.metadata['label']} - {doc.page_content}" for doc in docs]
    )


def generate_valid_prompt(prompt_template, max_tokens: int, tokenizer, **kwargs):
    """
    Generate a prompt that validates against a maximum token length. If the prompt exceeds the token limit,
    reduce the number of documents retrieved until the prompt fits.
    If it still exceeds the limit, include only 2 documents and print a warning.

    Args:
        prompt_template: The prompt template to format the final prompt.
        max_tokens: The maximum allowed token length for the prompt.
        tokenizer: The tokenizer used to compute token lengths.
        kwargs: Additional keyword arguments, such as title, description, and retrieved_docs.

    Returns:
        prompt (str): The final prompt that fits within the token limit.
    """

    def get_token_length(text):
        """Helper function to calculate token length of a given text."""
        return len(tokenizer.encode(text, truncation=False))

    # Extracting relevant fields from kwargs
    title = kwargs.get("title", "")
    description = kwargs.get("description", "")
    language = lang_mapping.loc[
        lang_mapping["lang_iso_2"] == kwargs.get("language", ""), "lang"
    ].values[0]
    retrieved_docs = kwargs.get("retrieved_docs", [])

    # Initialize prompt with all documents
    current_docs = retrieved_docs
    prompt = prompt_template.format(
        **{
            "title": title,
            "description": description,
            "language": language,
            "proposed_categories": format_docs(current_docs),
        }
    )

    # Check token length and adjust by reducing documents if necessary
    while get_token_length(prompt) > max_tokens and len(current_docs) > 0:
        # Remove the last document and retry
        current_docs = current_docs[:-1]
        prompt = prompt_template.format(
            **{
                "title": title,
                "description": description,
                "language": language,
                "proposed_categories": format_docs(current_docs),
            }
        )

    # If the prompt is still too long, include only 2 documents and print a warning
    if get_token_length(prompt) > max_tokens:
        current_docs = current_docs[:2]  # Include only 2 documents
        prompt = prompt_template.format(
            **{
                "title": title,
                "description": description,
                "language": language,
                "proposed_categories": format_docs(current_docs),
            }
        )
        print("Warning: The prompt is too long. Only 2 documents have been included.")
    num_documents_included = len(current_docs)
    return prompt, num_documents_included


def create_prompt_with_docs(row, parser, retriever, labels_en, **kwargs):
    description = getattr(row, kwargs.get("description_column"))
    title = getattr(row, kwargs.get("title_column"))
    lang = lang_mapping.loc[lang_mapping["lang_iso_2"] == row.lang, "lang"].values[0]
    id = row.id

    # Retrieve documents
    retrieved_docs = retriever.invoke(" ".join([title, description]))

    retrieved_codes = [doc.metadata["code"] for doc in retrieved_docs]
    relevant_code_en = labels_en[labels_en["code"].isin(retrieved_codes)].copy()
    relevant_code_en["code"] = pd.Categorical(
        relevant_code_en["code"], categories=retrieved_codes, ordered=True
    )
    relevant_code_en = relevant_code_en.sort_values("code")
    retrieved_docs_en = DataFrameLoader(relevant_code_en, page_content_column="description").load()

    # Generate the prompt and include the number of documents
    prompt_template = PromptTemplate.from_template(
        template=CLASSIF_PROMPT_TEMPLATE,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt = prompt_template.format(
        **{
            "title": title,
            "description": description,
            "language": lang,
            "proposed_categories": format_docs(retrieved_docs_en),
        }
    )

    return {"id": id, "prompt": prompt}


def create_translation_prompt(row, parser, **kwargs):
    description = getattr(row, kwargs.get("description_column"))
    title = getattr(row, kwargs.get("title_column"))
    lang = lang_mapping.loc[lang_mapping["lang_iso_2"] == row.lang, "lang"].values[0]

    template = EXTRACTION_PROMPT if lang == "English" else TRANSLATION_PROMPT
    return PromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    ).format(
        **{
            "title": title,
            "description": description,
            "language": lang,
        }
    )
