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


# Reply in a json format as shown in the example below, this is very important ! DO NOT ADD ANYTHING ELSE TO YOUR RESPONSE.

# Example format: {{"Code" : <your_code>, "Confidence" : <your_confidence>}}
