from src.utils.data import extract_info

from src.utils.mapping import lang_mapping

CLASSIF_PROMPT_SYS = """You are an expert in the International Standard Classification of Occupations (ISCO). Your goal is:

1. Analyze the job title and job description provided by the user.
2. From the list of occupational categories provided, identify the most appropriate ISCO code (4 digits) based on the job description. If the job description is not clear, use the job title to classify the job.
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
3. **Select relevant keywords**: Identify important keywords that classify the job offer, even if they are not directly mentioned in the text. For example, if the job is for a software developer, include terms like "programming" and "software development." Focus only on keywords related to the role and responsibilities, excluding irrelevant details like location or contract type.
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

1. **Extract and summarize** the relevant information about the **job title** and **job description**. Focus on key responsibilities, tasks, and role details, as this information will be used to classify the job offer.
2. Ensure the summary captures all key details and important keywords related to the job.
3. **Select relevant keywords**: Identify important keywords that classify the job offer, even if they are not directly mentioned in the text. For example, if the job is for a software developer, include terms like "programming" and "software development." Focus only on keywords related to the role and responsibilities, excluding irrelevant details like location or contract type.
4. **Return the job title, the summarized job description and keywords in a JSON format**, as specified by the user.

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
        [
            f"{doc.metadata['code']}: {doc.metadata['label']} - {extract_info(doc.page_content, paragraphs=["description", "examples"])}"
            for doc in docs
        ]
    )


def create_prompt_with_docs(row, parser, retriever, labels_en, **kwargs):
    task_description = "Retrieve the most relevant ISCO documents from the dataset based on the provided input by matching them with the closest ISCO classification labels."

    title = getattr(row, kwargs.get("title_column"))
    description = getattr(row, kwargs.get("description_column"))
    keywords = ", ".join(row.keywords.tolist()) if row.keywords is not None else None

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

    all_docs = filter(None, [docs_title, docs_description, docs_keywords])

    retrieved_docs = []
    for lst in all_docs:
        for item in lst:
            if item not in retrieved_docs:
                retrieved_docs.append(item)

    prompt = CLASSIF_PROMPT.format(
        **{
            "title": title,
            "description": description,
            "proposed_categories": format_docs(retrieved_docs),
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
