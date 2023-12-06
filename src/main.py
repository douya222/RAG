from llm import model_loader
from data_loader import txt_data, txt_split
from embedding import vectorize_documents, load_chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import json
from tqdm import tqdm

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
llm_model = "/data/datasets/user1801004151/model_weights/chatglm3-6b/" # llm model
LLM = model_loader(model_path=llm_model)

# construct prompt
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

# llm chain
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt 
    | LLM 
    | StrOutputParser()
)

# Q&A
qa_path = './qa_data/qa.json'
output_path = './qa_data/qa_p.json'

with open(qa_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

progress_bar = tqdm(total=len(data), desc="Processing")
for item in data:
    question = item["question"]
    item["predicted_llm"] = LLM(question)
    item["predicted_rag"] = rag_chain.invoke(question)
    progress_bar.update(1)

progress_bar.close()
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, indent=2, ensure_ascii=False)

print(f"File has been successfully processed and written to: {output_path}")
# print("==================chatglm3 only===================")
# print(LLM(query))
# print("==================retrieval + chatglm3===================")
# print(rag_chain.invoke(query))




