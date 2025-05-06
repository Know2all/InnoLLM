import os
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from config import Constant,logger
from langchain.prompts import PromptTemplate
from langchain_mongodb import MongoDBChatMessageHistory

class ZoyaChatbot:
    def __init__(self,prompt,**kwargs):
        logger.info("Zoya Bot Initialized")
        self.model = kwargs.get('model',None)
        self.prompt = prompt
        self.llm = kwargs.get('llm',Constant.LLM_GOOGLE)
        self.embeddings = kwargs.get('embeddings',HuggingFaceEmbeddings(model_name=Constant.DEFAULT_HUGG_MODEL))
        self.uid = kwargs.get('uid','self')
        self.history = MongoDBChatMessageHistory(
            connection_string=Constant.MONGO_URL,
            database_name='chatbot',
            collection_name='zoya',
            session_id=self.uid,
            history_size=3
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=self.history
        )
        self.vectorstore = self.load_model(model=self.model)
        self.build_chain()
    
    def load_model(self,model):
        logger.info("Model Loaded")
        if not os.path.exists(model):
            raise Exception("Model Not Found !")
        return Chroma(
            persist_directory=self.model,
            embedding_function=self.embeddings
        )

    def load_prompt(self):
        with open(self.prompt) as f:
            custom_prompt_template = f.read()
        self.qa_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        return self.qa_prompt

    def build_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, 
            retriever=retriever, 
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.load_prompt()},
            verbose=False
        )
    
    def invoke(self, query: str, uid: str = None):
        if uid:
            self.uid = uid
            self.history = MongoDBChatMessageHistory(
                connection_string=Constant.MONGO_URL,
                database_name='chatbot',
                collection_name='zoya',
                session_id=uid,
                history_size=3
            )
            self.memory.chat_memory = self.history
        logger.info("Agent Thinking !")
        result = self.qa_chain.invoke({'question': query})
        return result['answer']

    

if __name__ == "__main__":
    zoya = ZoyaChatbot(model='assets/models/zoya_6k_allmini',uid='anwar',prompt='prompts/2.txt')
    while True:
        user = input("You :")
        if user in ('exit','quit'):
            break
        bot = zoya.invoke(query=user)
        print("Bot :\n",bot)