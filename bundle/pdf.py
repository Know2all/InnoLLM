import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from config import Constant,logger
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
from langchain_mongodb import MongoDBChatMessageHistory
from langchain.prompts import PromptTemplate

class PDFBot:
    def __init__(self,uid):
        logger.info("PDF Bot Initialized")
        self.llm =ChatGoogleGenerativeAI(
            model=Constant.MODEL,
            temperature=Constant.TEMPERATURE,
            api_key = Constant.API_KEY,
            top_k = Constant.TOP_K
        )
        self.uid = uid
        self.qa = None
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=Constant.API_KEY)
        self.history = MongoDBChatMessageHistory(
            connection_string=Constant.MONGO_URL,
            database_name="chatbot",
            collection_name="pdf_qa",
            session_id=uid
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

    def load_documents(self, files: list[str], mode: str = 'layout') -> list[Document]:
        logger.info("Pdf loaded")
        all_docs = []
        for file in files:
            loader = PyPDFLoader(file_path=file, extraction_mode=mode)
            documents = loader.load()
            all_docs.extend(documents)
        return all_docs
    
    def prompt(self):
        with open('prompts/1.txt','r') as f:
            custom_prompt_template = f.read()
        self.qa_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        return self.qa_prompt
    
    def create_vectorstore(self, documents,store):
        logger.info("Vectorstore Created")
        vectorstore = Chroma.from_documents(
            documents=documents[:1],
            embedding=self.embeddings,
            persist_directory=f'store/{store}'
        )
        for doc in tqdm(documents[1:], desc="Creating Vector Store", unit='doc'):
            vectorstore.add_documents([doc])
        return vectorstore
    
    def get_vectorstore(self,store):
        logger.info("Vectore Store Loaded")
        if not os.path.exists(f"store/{store}"):
            raise Exception("Vectore Store Not Exists")
        return Chroma(persist_directory=f'store/{store}',embedding_function=self.embeddings)
        
    def build_qa_chain(self,store):
        vectorstore = self.get_vectorstore(store=store)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.qa = ConversationalRetrievalChain.from_llm(llm=self.llm, retriever=retriever, memory=self.memory,combine_docs_chain_kwargs={"prompt": self.prompt()})
    
    def invoke(self,query:str)->str:
        response = self.qa.invoke({'question':query})
        answer = response['chat_history'][-1].content
        return answer
    

if __name__ == "__main__":
    bot = PDFBot(uid='anwar')
    # docs = bot.load_documents(files=['assets/input/natwest1.pdf','assets/input/natwest2.pdf'])
    # store = bot.create_vectorstore(documents=docs)
    # print("store created")
    bot.build_qa_chain(store='unique')
    while True:
        user = input("You :")
        if user in ('exit','quit'):
            break
        answer = bot.invoke(query=user)
        print('Bot :\n',answer)