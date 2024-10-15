# classif-win-challenge
---
In the area of Web Intelligence, the **European Statistics Awards Programme** aims to discover promising methodologies for processing of content from the World Wide Web with the purpose to extract valuable data to be used for statistical and analytical purposes.

The **European Statistics Awards**  WEB INTELLIGENCE CLASSIFICATION OF OCCUPATIONS FOR ONLINE JOB ADVERTISEMENTS CHALLENGE (WI CLASSIFICATION CHALLENGE) will focus on developing approaches that learn how to assign a class label (from the known taxonomy) to job advertisements from a given dataset.

see https://statistics-awards.eu/competitions/12#learn_the_details

---

Classifying job offers into structured taxonomies, such as the International Standard Classification of Occupations (ISCO), is a complex task, especially when dealing with multilingual data. Traditional methods often rely on large, manually labeled datasets and language-specific rules, making them difficult to scale and maintain. However, advancements in large language models (LLMs) provide a more efficient solution through the approach of "leveraging LLMs for classification tasks."

Processing steps :

Our framework processes the input file in four key stages, each designed to streamline the classification of job advertisements ):

1. Data Cleaning and Enrichment: The first step involves cleaning the data to remove inconsistencies and irrelevant information. We also enrich the dataset by adding important variables, such as identifying the language in which each job advertisement is written.
The output dataset is saved as a parquet file in directory data/processed-data/wi_dataset_by_lang

2. Translation and Keyword Detection: In the second step, we handle job ads written in languages other than English. These are translated into English (using our LLM model) to ensure consistency and to enable the use of a specialized english-language vector database (see step 3) for more accurate retrieval of relevant job descriptions. Simultaneously, we conduct keywords detection to highlight essential terms relevant to job classification.
The output dataset is saved in directory data/processed-data/wi_dataset_by_lang_translated

3. Vector Database Creation and Prompt Generation: In this step, we build a Chroma vector database by embedding the explanatory notes from the ISCO classification system, chunked by ISCO code. We then perform a similarity search between the embeddings of the job title, description, and keywords, and the embeddings of the explanatory notes. The top 5 most similar ISCO codes are selected. Using these top 5 codes, we generate tailored prompts for the LLM, ensuring that the model has access to the most relevant labels and context for accurate job classification.
The output dataset is saved in directory data/processed-data/wi_dataset_by_lang_prompts

4. Prediction: In the final step, the LLM uses the generated prompt to predict the correct ISCO code from the top 5 codes selected in the previous phase. The model leveraging the context and relevant information embedded in the prompt, chooses the most appropriate ISCO code among the shortlisted options.
The output dataset is saved as a parquet file in directory data/processed-data/predictions_by_lang

---
## How to use our code

The input data files (`wi_datasets.csv` and `wi_labels.csv`) should be placed in the `data/raw-data` directory. 

Our code is written in Python, and before executing it, please run the command:  
```
pip install -r requirements.txt
```
The names of the code files are prefixed with numbers, which indicate the order in which they should be executed :
```
python 1-detect-language.py
python 2-translate-descriptions.py
python 3-build-vdb.py
python 4-classif-llm.py
```

Each of the 4 steps output its results in the directory `data/processed-data/` in parquet format.
Final stage produce the classification.csv file in the directory `submissions/` as well as the parquet file in `data/processed-data/`

---
## Model used

To translate title and description of job advertissements as well as predict its ISCO code we currently use Qwen2.5-32B-Instruct (see https://github.com/QwenLM/Qwen2.5)
