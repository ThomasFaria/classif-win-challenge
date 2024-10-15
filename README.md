# Web Intelligence Classification Challenge for Online Job Advertisements

## Overview

This repository contains the codebase for the **European Statistics Awards Programme's WEB INTELLIGENCE CLASSIFICATION OF OCCUPATIONS FOR ONLINE JOB ADVERTISEMENTS CHALLENGE (WI CLASSIFICATION CHALLENGE)**. The goal is to develop innovative approaches for classifying job advertisements into standardized taxonomies, specifically the International Standard Classification of Occupations (ISCO).

## Challenge Description

The European Statistics Awards Programme aims to discover promising methodologies for processing content from the World Wide Web to extract valuable data for statistical and analytical purposes. This challenge focuses on developing approaches that learn how to assign class labels (from a known taxonomy) to job advertisements in a given dataset.

For more details, visit the [official challenge page](https://statistics-awards.eu/competitions/12#learn_the_details).

## Our Approach

We leverage advanced Large Language Models (LLMs) to tackle the complex task of classifying multilingual job offers into structured taxonomies. Our approach offers a more efficient solution compared to traditional methods that rely on large, manually labeled datasets and language-specific rules.

### Processing Pipeline

Our framework processes the input data in four key stages:

1. **Language Detection**
   - Detect the language of each job advertisement
   - Script: `1-detect-language.py`
   - Output: Parquet files in `data/processed-data/wi_dataset_by_lang`

2. **Translation**
   - Translate non-English job ads to English using our LLM model
   - Script: `2-translate-descriptions.py`
   - Output: Parquet files in `data/processed-data/wi_dataset_by_lang_translated`

3. **Vector Database Creation and Prompt Generation**
   - Build a Chroma vector database by embedding explanatory notes from the ISCO classification system
   - Generate tailored prompts for the LLM using the top 5 most similar ISCO codes
   - Script: `3-build-vdb.py`
   - Output: Parquet files in `data/processed-data/wi_dataset_by_lang_prompts`

4. **Prediction**
   - Use the LLM with generated prompts to predict the correct ISCO code
   - Script: `4-classif-llm.py`
   - Output: Parquet files in `data/processed-data/predictions_by_lang`

## Installation and Usage

1. Clone this repository:
   ```
   git clone https://github.com/thomasfaria/classif-win-challenge.git
   cd classif-win-challenge
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place the input data files (`wi_dataset.csv` and `wi_labels.csv`) in the `data/raw-data` directory.

4. Run the processing pipeline in order:
   ```
   python 1-detect-language.py
   python 2-translate-descriptions.py
   python 3-build-vdb.py
   python 4-classif-llm.py
   ```

## Argo Workflows

The project includes Argo Workflow configurations for orchestrating the pipeline in a Kubernetes environment. These workflows automate the execution of our classification pipeline, making it easier to process large datasets efficiently.

### Workflow Files

The workflow configuration files are located in the `argo-workflows` directory:

- `1-detect-lang.yaml`: Language detection step
- `2-translate-desc.yaml`: Translation step
- `3-create_prompts.yaml`: Vector database creation and prompt generation
- `4-make_predictions.yaml`: ISCO code prediction
- `pipeline_full.yml`: Combines all steps into a single workflow

### Running the Pipeline

To run the full pipeline using Argo Workflows, use the following command:

```bash
argo submit argo-workflows/pipeline_full.yml
```

### Important Note on Argo Workflows Execution

Running Argo Workflows requires access to a Kubernetes cluster with Argo installed. In our setup, we use the SSP Cloud environment, which provides the necessary infrastructure.

If you're interested in trying out the Argo Workflows for this project you'll need an SSP Cloud account. We would be happy to introduce you to the SSP Cloud environment and help you get started, so, please, reach out to us for assistance in setting up and running the workflows.


## Project Structure

```
.
├── 1-detect-language.py
├── 2-translate-descriptions.py
├── 3-build-vdb.py
├── 4-classif-llm.py
├── argo-workflows/
├── data/
│   ├── chroma_db/
│   ├── processed-data/
│   ├── raw-data/
│   └── submission/
├── Dockerfile
├── README.md
├── requirements.txt
└── src/
    ├── constants/
    ├── detect_lang/
    ├── llm/
    ├── prompting/
    ├── response/
    ├── utils/
    └── vector_db/
```

### Key Directories and Files

- `src/`: Contains the core Python modules for each component of the pipeline
- `data/`: Stores raw input data, processed data, and the Chroma vector database
- `argo-workflows/`: Contains Argo Workflow configuration files (templates)
- `Dockerfile`: For containerizing the application
- `requirements.txt`: Lists Python dependencies

## Model Information

We currently use Qwen2.5-32B-Instruct (https://github.com/QwenLM/Qwen2.5) for translation and ISCO code prediction. This model requires approximately 75GB of VRAM. If you don't have access to such hardware, consider using smaller models.

## License

This project is licensed under the terms of the LICENSE file in the root directory.
