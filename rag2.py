from langchain.document_loaders import TextLoader
import argparse
import logging
from dingtalk_stream import AckMessage
import dingtalk_stream
from langchain.prompts import ChatPromptTemplate
import os
import dashscope
from dashscope import TextEmbedding
from dashvector import Client, Doc
from dashvector import Doc
import dashvector
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import sys

from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain
import torch
loader = TextLoader("./data/市场监督实务培训-test.txt")
documents = loader.load()

# 文档分割
from langchain.text_splitter import CharacterTextSplitter
# 创建拆分器
text_splitter = CharacterTextSplitter(separator='。', chunk_size=512, chunk_overlap=0)
# 拆分文档
documents = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

# embedding model: m3e-base
model_name = "/data/datasets/user1801004151/model_weights/m3e-base"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="为文本生成向量表示用于文本检索"
            )

# load data to Chroma db
db = Chroma.from_documents(documents, embedding)
# similarity search
# db.similarity_search("藜一般在几月播种？")

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
prompt1 = PromptTemplate(template=template, input_variables=["context", "question"])

from langchain import LLMChain
from langchain_wenxin.llms import Wenxin
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.llms.base import LLM
class GLM(LLM):
    max_token: int = 2048
    temperature: float = 0.001
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    
    history_len: int = 1024
    
    def __init__(self):
        super().__init__()
        
    @property
    def _llm_type(self) -> str:
        return "GLM"
            
    def load_model(self, llm_device="gpu",model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True).half().cuda()

    def _call(self,prompt:str,history:List[str] = [],stop: Optional[List[str]] = None):
        response, _ = self.model.chat(
                    self.tokenizer,prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,temperature=self.temperature,
                    top_p=self.top_p)
        return response

modelpath = "/data/datasets/user1801004151/model_weights/chatglm3-6b/"
# LLM选型
llm = GLM()
llm.load_model(model_name_or_path = modelpath)

prompt = ChatPromptTemplate.from_template(template)
retriever = db.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
# qa({"question": "藜发生虫害的防治方法？？"})
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)
query = '地方性法规可以设定哪些行政处罚？'
rag_chain.invoke(query)
print(llm(query))
print(rag_chain.invoke(query))

