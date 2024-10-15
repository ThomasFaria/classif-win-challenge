import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import PydanticOutputParser

from src.utils.data import extract_info
from src.utils.mapping import lang_mapping

# System prompt for ISCO (International Standard Classification of Occupations) classification.
# This prompt instructs the model to classify job descriptions or titles into ISCO codes.
CLASSIF_PROMPT_SYS = """You are an expert in the International Standard Classification of Occupations (ISCO). Your goal is:

1. Analyze the job title and job description provided by the user.
2. From the list of occupational categories provided, identify the most appropriate ISCO code (4 digits) based on the job description. If the job description is not clear, use the job title to classify the job.
3. Return the 4-digit code in JSON format as specified by the user. If the job cannot be classified within the given categories, return `null` in the JSON.
"""

# User prompt for ISCO classification.
# This prompt includes job title, description, and relevant occupational categories.
CLASSIF_PROMPT = """\
- Job Ad Title:
{title}

- Job Ad Description:
{description}

- Relevant Occupational Categories:
{proposed_categories}

{format_instructions}
"""

# System prompt for translation tasks.
# It instructs the model to translate job ads from a specified language into English.
TRANSLATION_PROMPT_SYS = """You are an excellent translator from {language} to English. You will be provided with a possibly noisy job offer text in {language} that you have to translate. Your task is:

1. **Extract and translate the job title** from {language} to English. Ensure that the title is accurately translated and meaningful.
2. **Extract and translate the job description** (which includes key responsibilities, tasks, and role details) from {language} to English. Summarize the description if necessary, but ensure that all relevant keywords and key information are included. Usually three or four sentences are enough.
3. **Select relevant keywords**: Identify important keywords that classify the job offer, even if they are not directly mentioned in the text. For example, if the job is for a software developer, include terms like "programming" and "software development." Focus only on keywords related to the role and responsibilities, excluding irrelevant details like location or contract type.
4. **Return the translation in JSON format** as explained by the user.
"""

# User prompt for translation.
# This format includes the job title and description to be translated into English.
TRANSLATION_PROMPT = """\
- Translate the following Job Ad Title in english:
{title}

- Translate the following Job Ad Description in english:
{description}

{format_instructions}
"""

# System prompt for information extraction.
# Instructs the model to extract and summarize job information from a possibly noisy job ad.
EXTRACTION_PROMPT_SYS = """You are a specialist in summarization. You will be provided with a possibly noisy job offer, and your task is to extract the main information. Specifically:

1. **Extract and summarize** the relevant information about the **job title** and **job description**. Focus on key responsibilities, tasks, and role details, as this information will be used to classify the job offer.
2. Ensure the summary captures all key details and important keywords related to the job.
3. **Select relevant keywords**: Identify important keywords that classify the job offer, even if they are not directly mentioned in the text. For example, if the job is for a software developer, include terms like "programming" and "software development." Focus only on keywords related to the role and responsibilities, excluding irrelevant details like location or contract type.
4. **Return the job title, the summarized job description and keywords in a JSON format**, as specified by the user.
"""

# User prompt for extraction.
# This format includes the job title and description for summarization and keyword extraction.
EXTRACTION_PROMPT = """\
- Job Ad Title:
{title}

- Job Ad Description:
{description}

{format_instructions}
"""


# Function to format documents retrieved from a dataset.
# The function extracts specific information from the documents (based on paragraphs like 'description' and 'examples').
def format_docs(docs: list):
    """
    Format the retrieved documents to be included in the prompt.

    Parameters:
    ----------
    docs : list
        A list of documents retrieved from the dataset.

    Returns:
    -------
    str
        A formatted string containing the document metadata, label, and extracted information.
    """
    return "\n\n".join(
        [
            f"{doc.metadata['code']}: {doc.metadata['label']} - {extract_info(doc.page_content, paragraphs=['description', 'examples'])}"
            for doc in docs
        ]
    )


# Function to create a prompt with relevant ISCO documents for classification.
# It retrieves the most relevant documents for classification based on the job title, description, or keywords.
def create_prompt_with_docs(
    row: pd.Series, parser: PydanticOutputParser, retriever: Chroma, **kwargs
):
    """
    Create a prompt for classifying job descriptions into ISCO codes. The prompt includes the job title, description, and relevant ISCO documents.

    Parameters:
    ----------
    row : pd.Series
        The row from the dataset containing job details.
    parser : PydanticOutputParser
        The parser for handling model responses.
    retriever : Chroma
        The retriever for fetching relevant ISCO documents.
    kwargs : dict
        Additional keyword arguments.

    Returns:
    -------
    list
        A list of prompts in conversational format for the classification task.
    """

    task_description = "Retrieve the most relevant ISCO documents from the dataset based on the provided input by matching them with the closest ISCO classification labels."

    # Extract job title, description, and keywords from the dataset row.
    title = getattr(row, kwargs.get("title_column"))
    description = getattr(row, kwargs.get("description_column"))
    keywords = ", ".join(row.keywords.tolist()) if row.keywords is not None else None

    # Create a query to retrieve documents related to the job title, description, or keywords.
    input_txt = "Instruct: {task_description}\nQuery: {query}"

    docs_title = (
        retriever.invoke(input_txt.format(task_description=task_description, query=title))
        if title is not None
        else None
    )
    docs_description = (
        retriever.invoke(input_txt.format(task_description=task_description, query=description))
        if description is not None
        else None
    )
    docs_keywords = (
        retriever.invoke(input_txt.format(task_description=task_description, query=keywords))
        if keywords is not None
        else None
    )

    # Combine all retrieved documents, ensuring no duplicates.
    all_docs = filter(None, [docs_title, docs_description, docs_keywords])

    retrieved_docs = []
    for lst in all_docs:
        for item in lst:
            if item not in retrieved_docs:
                retrieved_docs.append(item)

    # Generate the classification prompt by inserting job details and relevant ISCO documents.
    prompt = CLASSIF_PROMPT.format(
        **{
            "title": title,
            "description": description,
            "proposed_categories": format_docs(retrieved_docs),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    # Return the complete prompt for classification in a conversational format (system role and user role).
    return [{"role": "system", "content": CLASSIF_PROMPT_SYS}, {"role": "user", "content": prompt}]


# Function to create a translation prompt based on the job ad language.
# It either uses the extraction prompt (for English) or translation prompt (for other languages).
def create_translation_prompt(row: pd.Series, parser: PydanticOutputParser, **kwargs):
    """
    Create a prompt for translating job descriptions from various languages into English.

    Parameters:
    ----------
    row : pd.Series
        The row from the dataset containing job details.
    parser : PydanticOutputParser
        The parser for handling model responses.
    kwargs : dict
        Additional keyword arguments.

    Returns:
    -------
    list
        A list of prompts in conversational format for the translation task.
    """

    description = getattr(row, kwargs.get("description_column"))
    title = getattr(row, kwargs.get("title_column"))
    lang = lang_mapping.loc[lang_mapping["lang_iso_2"] == row.lang, "lang"].values[0]

    # Choose the prompt template based on the language (English uses extraction, other languages use translation).
    template = EXTRACTION_PROMPT if lang == "English" else TRANSLATION_PROMPT
    template_sys = EXTRACTION_PROMPT_SYS if lang == "English" else TRANSLATION_PROMPT_SYS

    # Format the prompt with job title, description, and translation/extraction instructions.
    prompt = template.format(
        **{
            "title": title,
            "description": description,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    # Return the prompt in conversational format for the translation or extraction task.
    return [
        {"role": "system", "content": template_sys.format(language=lang)},
        {"role": "user", "content": prompt},
    ]
