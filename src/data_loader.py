from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

data_path = "../data/"

# 数据加载
def txt_data(txt_name):
    loader = TextLoader(data_path + txt_name)
    documents = loader.load()
    return documents

# 文本分割
def txt_split(documents):
    # 创建拆分器
    # text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=0) #只分割成一段
    text_splitter = CharacterTextSplitter(separator='。', chunk_size=128, chunk_overlap=0)
    # 拆分文档
    documents = text_splitter.split_documents(documents)
    return documents

def save_to_file(split_doc, file_name):
    # 生成保存分割文本的文件夹
    split_data_path = "../split_data"
    if not os.path.exists(split_data_path):
        os.mkdir(split_data_path)
    # 保存每个分割后的字符串为一个txt文件  
    with open(os.path.join(split_data_path, file_name), "w") as f: 
        for doc in split_doc:
            text = str(doc)
            f.write(text + "\n")


# doc = txt_data("藜.txt")
# doc_split = txt_split(doc)
# save_to_file(doc_split, "藜s.txt")


