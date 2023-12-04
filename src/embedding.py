from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from data_loader import txt_data, txt_split
import json
import os

def get_embedding(model_name):
    # get embedding model
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    query_instruction="为文本生成向量表示用于文本检索"
                )
    return embedding

def vectorize_documents(model_name, documents, db_path):
    # first vectorize and load data to Chroma db
    embedding = get_embedding(model_name)
    db = Chroma.from_documents(documents, embedding, persist_directory=db_path)
    db.persist()
    return db

def load_chroma(persist_directory):
    # load exit persisted Chroma db
    embedding = get_embedding(model_name)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return vectordb


doc = txt_data("市场监督实务培训-test.txt")
documents = txt_split(doc) 
db_path = "../database" # 数据库保存路径
model_name = "/data/datasets/user1801004151/model_weights/m3e-base" # m3a-base model
# embedding = get_embedding(model_name)
db = vectorize_documents(model_name, documents, db_path) # 首次向量化数据
db = load_chroma(persist_directory=db_path) # 加载已有向量数据库
# res = db.similarity_search("藜怎么防治虫害？")
# data = db.get() 
# with open("all_data.json", "w", encoding="utf-8") as f:
#     for i in range(len(data["ids"])):
#         item = {
#             "id": data["ids"][i],
#             "metadata": data["metadatas"][i], 
#             "document": data["documents"][i]
#         }
        
#         json.dump(item, f, ensure_ascii=False)
#         f.write("\n")
