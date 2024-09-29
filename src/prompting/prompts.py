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
2. **Extract and translate the job description** (which includes key responsibilities, tasks, and role details) from {language} to English. Summarize the description if necessary, but ensure that all relevant keywords and key information are included. Usually three or four sentences are enough.
3. **Return the translation in JSON format** as explained by the user.
"""

TRANSLATION_PROMPT = """
- Translate the following Job Ad Title in english:
{title}

- Translate the following Job Ad Description in english:
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


def create_prompt_with_docs(row, parser, retriever, labels_en, **kwargs):
    description = getattr(row, kwargs.get("description_column"))
    title = getattr(row, kwargs.get("title_column"))
    keywords = ", ".join(row.keywords.tolist()) if row.keywords is not None else None
    
    query = (
        "\n".join(filter(None, [title, keywords, description]))
        if title or description
        else "undefined"
    )

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
