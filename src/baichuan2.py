from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
import json
from tqdm import tqdm
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from data_loader import txt_data, txt_split
from embedding import vectorize_documents, load_chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# model_path = "/data/datasets/user1801004151/model_weights/Baichuan2-7B-Chat/"
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained(model_path)
# messages = []
# messages.append({"role": "user", "content": "解释一下“温故而知新”"})
# print(model.chat(tokenizer, messages))

class Baichuan(LLM):
    # baichuan model init
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
        return "baichuan2"
            
    def load_model(self, llm_device="cuda",model_name_or_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto',trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    
    def _call(self,prompt:str,history:List[str] = [],stop: Optional[List[str]] = None):
        messages = []
        history=history[-self.history_len:] if self.history_len > 0 else []
        messages.append({"role": "user", "content": "".join(history)+prompt})
        response = self.model.chat(
                    self.tokenizer,messages)
        return response

# model_path = "/data/datasets/user1801004151/model_weights/Baichuan2-7B-Chat/"
# llm = Baichuan()
# llm.load_model(model_name_or_path = model_path)
