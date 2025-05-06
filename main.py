from fastapi import FastAPI
from fastapi.responses import JSONResponse
from bundle.schema import PdfInput,CsvInput,MongoInput,KnowledgeInput
from bundle.pdf import PDFBot
from bundle.csv import PandasAgent
from bundle.memory import ConversationMemory
from bundle.mongodb import MongoDBClient
from bundle.agents import MongoBot
from config import *
from fastapi.requests import Request
from fastapi import UploadFile, File,Form
from typing import List,Optional
from pathlib import Path
from agent import APIToolBuilder
import tempfile

app = FastAPI(
    title="Innobot Web API",
    description="Backend for InnoBot",
    version="1.0.0",
)

@app.get("/api/hello")
async def hello():
    return JSONResponse(status_code=200,content={'message':'hello'})

@app.post('/api/agent')
async def chat(vectorDB: Optional[str] = Form(None), question: str = Form(...), uid: str = Form(...)):
    try:
        prompt = f"""Tries to provide an answer based on your tools with the following piece of information.
question: {question}
uid: {uid}
vectorDB: {vectorDB}
"""
        model_config = {
            "model": Constant.MODEL,
            "temperature": Constant.TEMPERATURE,
            "api_key": Constant.API_KEY,
            "max_tokens": Constant.MAX_TOKENS,
            "top_k": Constant.TOP_K,
        }
        tool_builder = APIToolBuilder(routes=Constant.ROUTES, model_config=model_config)
        tool_builder.stream(prompt)
        return JSONResponse(status_code=200, content="")
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
    
@app.post('/api/chat-pdf')
async def chat_with_pdf(pdfInput:PdfInput):
    try:
        pdfBot = PDFBot(uid=pdfInput.uid)
        pdfBot.build_qa_chain(store = pdfInput.vectorDB)
        answer = pdfBot.invoke(query=pdfInput.question)
        return JSONResponse(status_code=200,content={'answer':answer})
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(status_code=500,content={'error':str(e)})

@app.post('/api/chat-csv')
async def chat_with_csv(csvInput:CsvInput):
    try:
        memory = ConversationMemory(db_client=MongoDBClient(uri=Constant.MONGO_URL,db_name="chatbot"),collection_name="csv_qa")
        pandasBot = PandasAgent(csv_file=csvInput.file,uid=csvInput.uid,conversation_memory=memory)
        answer = pandasBot.run(question=csvInput.question)
        return JSONResponse(status_code=200,content={'answer':answer})
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(status_code=500,content={'error':str(e)})

@app.post('/api/chat-mongo')
async def chat_with_mongo(mongoInput: MongoInput):
    try:
        mongoBot = MongoBot(connection_string=mongoInput.mongo_uri, db_name=mongoInput.db_name)
        answer = mongoBot.run(question=mongoInput.question)
        return JSONResponse(status_code=200, content={'answer': answer})
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.post('/api/upload')
async def upload(files: List[UploadFile] = File(...),uid:str = Form(...)):
    try:
        uploaded_files = []
        for file in files:
            ext = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(await file.read())
                temp_file = tmp.name
                uploaded_files.append(temp_file)
        pdfbot = PDFBot(uid=uid)
        docs = pdfbot.load_documents(files=uploaded_files)
        store = pdfbot.create_vectorstore(documents=docs,store='unique')
        return JSONResponse(status_code=200,content={'store':'store/unique'})
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(status_code=500, content={'error': str(e)})
    
@app.post('/api/knowledge')
async def chat_with_knowlegde(knowledgeInput:KnowledgeInput):
    try:
        pdfBot = PDFBot(uid=knowledgeInput.uid)
        pdfBot.build_qa_chain(store = 'k1')
        answer = pdfBot.invoke(query=knowledgeInput.question)
        return JSONResponse(status_code=200,content={'answer':answer})
    except Exception as e:
        return JSONResponse(status_code=500,content={'error':str(e)})