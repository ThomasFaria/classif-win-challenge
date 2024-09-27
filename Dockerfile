FROM inseefrlab/onyxia-vscode-pytorch:py3.12.5

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt