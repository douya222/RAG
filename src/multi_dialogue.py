from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from llm import model_loader
from llm import model_loader
from data_loader import txt_data, txt_split
from embedding import vectorize_documents, load_chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import json
from tqdm import tqdm
from baichuan2 import Baichuan
from llama2 import Llama2
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

template = '''
        【任务描述】
        请根据用户输入的上下文回答问题，并遵守回答要求。

        【背景知识】
        {context}

        【回答要求】
        - 你需要严格根据背景知识的内容回答，禁止根据常识和已知信息回答问题。
        - 对于不知道的信息，直接回答“未找到相关答案”
        -----------
        {question}
        '''


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

# load llm
# chatglm3-6b/chatglm2-6b/Baichuan2-7B-Chat/Llama-2-7b-chat-hf
llm_model = "/data/datasets/user1801004151/model_weights/chatglm3-6b/" # llm model
llm = model_loader(model_path=llm_model)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    output_key='answer',
    memory_key='chat_history',
    return_messages=True)


messages = [
  SystemMessagePromptTemplate.from_template(template),
  HumanMessagePromptTemplate.from_template('{question}')
]
prompt = ChatPromptTemplate.from_messages(messages)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    get_chat_history=lambda h : h,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=False)

# 多轮对话：保持历史记录
response = chain({"question": "藜麦怎么防治虫害？"})
print(f"{response['chat_history']}\n")

response = chain({"question": "那怎么防治病害呢？"})
print(f"{response['chat_history']}\n")

# response = chain({"question": "藜麦怎么防治虫害？"})
# print(f"{response['chat_history']}\n")