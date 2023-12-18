from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from data_loader import txt_data, txt_split
from embedding import vectorize_documents, load_chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import json
from tqdm import tqdm
from baichuan2 import Baichuan
cache_instance = InMemoryCache()

set_llm_cache(cache_instance)
cache_instance = InMemoryCache()
set_llm_cache(cache_instance)

def pipeline():

    # load db
    db_path = "../database" # 数据库保存路径
    # new db
    doc = txt_data("藜.txt")
    documents = txt_split(doc) 
    embedding_model = "/data/datasets/user1801004151/model_weights/m3e-base" # m3a-base model
    db = vectorize_documents(embedding_model, documents, db_path)
    # db existed
    # db = load_chroma(persist_directory=db_path)
    retriever = db.as_retriever()
  
    template = '''
        【任务描述】
        请根据用户输入的上下文回答问题，并遵守回答要求。

        【背景知识】
        {context}

        【回答要求】
        - 你需要严格根据背景知识的内容回答，禁止根据常识和已知信息回答问题。
        - 对于不知道的信息，直接回答“未找到相关答案”
        - 可以根据背景知识进行总结
        -----------
        {question}
        '''

    prompt = ChatPromptTemplate.from_template(template)
    
    chain_type_kwargs = {"prompt": prompt}
    
    model_path = "/data/datasets/user1801004151/model_weights/Baichuan2-7B-Chat/"
    LLM = Baichuan()
    LLM.load_model(model_name_or_path = model_path)
    qa = RetrievalQA.from_chain_type(
        llm=LLM, 
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents= True,
    )
    
    res = qa({"query":"藜麦的穗部可以呈现哪几种颜色"})
    print(res)
    print(cache_instance._cache)

pipeline()