from langchain_core.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = """You are tasked with helping classify job advertisements into predefined occupational categories.

Given a job advertisement description and a list of the most relevant occupational categories, your goal is to select the single most appropriate occupational category for the job based on the description and its title.

Job Ad Title:
{title}

Job Ad Description:
{description}

Relevant Occupational Categories:
{proposed_categories}

Please choose the best occupational category based on the description and give a likelihood estimate between 0 and 1 for your confidence.
{format_instructions}
"""


def format_docs(docs: list):
    return "\n\n".join(
        [
            f"{i + 1}. {doc.metadata['code']} - {doc.metadata['label']}: {doc.page_content}"
            for i, doc in enumerate(docs)
        ]
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
    retrieved_docs = kwargs.get("retrieved_docs", [])

    # Initialize prompt with all documents
    current_docs = retrieved_docs
    prompt = prompt_template.format(
        **{
            "title": title,
            "description": description,
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
                "proposed_categories": format_docs(current_docs),
            }
        )
        print("Warning: The prompt is too long. Only 2 documents have been included.")
    num_documents_included = len(current_docs)
    return prompt, num_documents_included


def create_prompt_with_docs(row, parser, tokenizer, retriever, **kwargs):
    description = getattr(row, kwargs.get("description_column"))
    title = getattr(row, kwargs.get("title_column"))
    id = row.id

    # Retrieve documents and make sure a document is not retrieved twice
    retrieved_docs = retriever.invoke(" ".join([title, description]))
    retrieved_docs_unique = []
    for item in retrieved_docs:
        if item not in retrieved_docs_unique:
            retrieved_docs_unique.append(item)

    # Generate the prompt and include the number of documents
    prompt_template = PromptTemplate.from_template(
        template=RAG_PROMPT_TEMPLATE,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt, _ = generate_valid_prompt(
        prompt_template,
        kwargs.get("prompt_max_token"),
        tokenizer,
        title=title,
        description=description,
        retrieved_docs=retrieved_docs_unique,
    )

    return {"id": id, "prompt": prompt}
