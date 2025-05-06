import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Constant,logger
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

class RAG:
    def __init__(self,embeddings):
        logger.info("Rag initalizeed")
        self.embeddings = embeddings

    def load_documents(self, files: list[str], mode: str = 'layout') -> list[Document]:
        logger.info("Documents loaded")
        all_docs = []
        for file in files:
            loader = PyPDFLoader(file_path=file, extraction_mode=mode)
            documents = loader.load()
            all_docs.extend(documents)
        return all_docs
    
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
    
if __name__ == "__main__":
    embeddings = GoogleGenerativeAIEmbeddings(model=Constant.EMBEDDING_MODEL,google_api_key=Constant.API_KEY)
    rag = RAG(embeddings=embeddings)
    docs = rag.load_documents(files=[ os.path.join('knowledge/',file)  for file in os.listdir('knowledge')])
    store = rag.create_vectorstore(documents=docs,store='k1')