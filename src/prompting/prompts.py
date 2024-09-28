import pandas as pd

from langchain_community.document_loaders import DataFrameLoader

from src.utils.mapping import lang_mapping

CLASSIF_PROMPT_SYS = """Your are an expert in the International Standard Classification of Occupations (ISCO). Your goal is:
1. Analyse the job title and the job description given by the user.
2. Find the most appropriate code from the list of a relevant occupational categories given by the user.
3. Return the code (4 digits) in a json format as explained by the user. If the job description can not be classified into the proposed categories, return None in the json.
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
1. Extract only the relevant information about the job title and the job description. Focus solely on the content related to the job description, which includes the key responsibilities, tasks, and role details, so that the information can later be used to classify the job offer.
2. Translate into English only the extracted informations from the description. You can summarise it, be but sure to add the key words of the job described.
3. Translate into English the job title exactly as it appears.
4. Reply in a json format as explained by the user.
"""

TRANSLATION_PROMPT = """
- Job Ad Title (to translate exactly in english):
{title}

- Job Ad Description (to extract and translate in enlish):
{description}

{format_instructions}
"""

EXTRACTION_PROMPT_SYS = """You are a specialist in summarization. You will be provided with a possibly noisy job offer that you will have to extract the main informations. Your task is:
1. Extract only the relevant information about the job title and the job description. Focus solely on the content related to the job description, which includes the key responsibilities, tasks, and role details, so that the information can later be used to classify the job offer.
2. You can summarise it, be but sure to add the key words of the job described.
3. Return the job title and the job description in a json format as explained by the user.
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
    description = description if description is not None else getattr(row, "description_clean")
    title = getattr(row, kwargs.get("title_column"))
    title = title if title is not None else getattr(row, "title_clean")

    # Retrieve documents
    retrieved_docs = retriever.invoke(" ".join([title, description]))

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
            "proposed_categories": retrieved_docs_en,
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
