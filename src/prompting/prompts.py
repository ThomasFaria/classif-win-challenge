import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

from src.utils.mapping import lang_mapping

CLASSIF_PROMPT_SYS = """You are an expert in the International Standard Classification of Occupations (ISCO). Your goal is:

1. Analyze the job title and job description provided by the user.
2. From the list of occupational categories provided, identify the most appropriate ISCO code (4 digits) based on the job description.
3. Return the 4-digit code in JSON format as specified by the user. If the job cannot be classified within the given categories, return `null` in the JSON.
"""

CLASSIF_PROMPT = """
- Job Ad Title:
{title}

- Job Ad Description:
{description}

- Relevant Occupational Categories:
{proposed_categories}

{format_instructions}
"""

TRANSLATION_PROMPT_SYS = """You are an excellent translator from {language} to English. You will be provided with a possibly noisy job offer text in {language} that you have to translate. Your task is:

1. **Extract and translate the job title** from {language} to English. Ensure that the title is accurately translated and meaningful.
2. **Extract and translate the job description** (which includes key responsibilities, tasks, and role details) from {language} to English. Summarize the description if necessary, but ensure that all relevant keywords and key information are included.
3. **Return the translation in JSON format** as explained by the user.
"""

TRANSLATION_PROMPT = """
- Job Ad Title (to translate exactly in english):
{title}

- Job Ad Description (to extract and translate in enlish):
{description}

{format_instructions}
"""

EXTRACTION_PROMPT_SYS = """You are a specialist in summarization. You will be provided with a possibly noisy job offer, and your task is to extract the main information. Specifically:

1. Extract and summarize the relevant information about the **job title** and **job description**. Focus on key responsibilities, tasks, and role details, as this information will be used to classify the job offer.
2. Ensure the summary captures all key details and important keywords related to the job.
3. Return both the job title and the summarized job description in a JSON format, as specified by the user.

"""

EXTRACTION_PROMPT = """
- Job Ad Title:
{title}

- Job Ad Description:
{description}

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
    query = " ".join(filter(None, [title, description])) if title or description else "undefined"

    # Retrieve documents
    retrieved_docs = retriever.invoke(query)

    retrieved_codes = [doc.metadata["code"] for doc in retrieved_docs]
    relevant_code_en = labels_en[labels_en["code"].isin(retrieved_codes)].copy()
    relevant_code_en["code"] = pd.Categorical(
        relevant_code_en["code"], categories=retrieved_codes, ordered=True
    )
    relevant_code_en = relevant_code_en.sort_values("code")
    retrieved_docs_en = DataFrameLoader(relevant_code_en, page_content_column="description").load()

    prompt = CLASSIF_PROMPT.format(
        **{
            "title": title,
            "description": description,
            "proposed_categories": format_docs(retrieved_docs_en),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return [{"role": "system", "content": CLASSIF_PROMPT_SYS}, {"role": "user", "content": prompt}]


def create_translation_prompt(row, parser, **kwargs):
    description = getattr(row, kwargs.get("description_column"))
    title = getattr(row, kwargs.get("title_column"))
    lang = lang_mapping.loc[lang_mapping["lang_iso_2"] == row.lang, "lang"].values[0]

    template = EXTRACTION_PROMPT if lang == "English" else TRANSLATION_PROMPT
    template_sys = EXTRACTION_PROMPT_SYS if lang == "English" else TRANSLATION_PROMPT_SYS

    prompt = template.format(
        **{
            "title": title,
            "description": description,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return [
        {"role": "system", "content": template_sys.format(language=lang)},
        {"role": "user", "content": prompt},
    ]
