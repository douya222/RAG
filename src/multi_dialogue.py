from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from llm import model_loader
from embedding import vectorize_documents, load_chroma
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

# 构建初始 messages 列表，这里可以理解为是 openai 传入的 messages 参数
messages = [
  SystemMessagePromptTemplate.from_template(template),
  HumanMessagePromptTemplate.from_template('{question}')
]
# load llm
llm_model = "/data/datasets/user1801004151/model_weights/chatglm3-6b/" # llm model
LLM = model_loader(model_path=llm_model)
# load db
db_path = "../database" # 数据库保存路径
db = load_chroma(persist_directory=db_path)
retriever = db.as_retriever()

# 初始化 prompt 对象
prompt = ChatPromptTemplate.from_messages(messages)
llm_chain = LLMChain(llm=LLM, prompt=prompt)

combine_docs_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_separator="\n\n",
    document_variable_name="context",
)
q_gen_chain = LLMChain(llm=LLM, prompt=PromptTemplate.from_template(template))

qa = ConversationalRetrievalChain(combine_docs_chain=combine_docs_chain,
                                  question_generator=q_gen_chain,
                                  return_source_documents=True,
                                  return_generated_question=True,
                                  retriever=retriever)
print(qa({'question': "藜麦怎么防治虫害？", "chat_history": []}))
# Todo:
# 变成多轮对话，并存入对话历史