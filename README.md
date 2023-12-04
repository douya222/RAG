# Retrieval Augmented Generation(RAG)
m3e embedding model + chroma db + chatglm3 + langchain

## data
Put your data(.txt) into the RAG/data directory, see src/data_loader.py and src/embedding.py for details on data processing.

## model
embedding model: [m3e-base download](https://huggingface.co/moka-ai/m3e-base)

LLM model: [chatglm3-6b download](https://huggingface.co/THUDM/chatglm3-6b)

## conda env
```
conda create -n RAG python=3.10
source activate RAG
pip install -r requirements
```
## Todo
- More data formats(xlsx,json,pdf...)
- multi round of dialogue


