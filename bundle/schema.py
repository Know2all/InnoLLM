from pydantic import HttpUrl,BaseModel

class InstaLink(BaseModel):
    link: HttpUrl

class ZoyaInput(BaseModel):
    uid:int
    question:str

class PdfInput(BaseModel):
    question:str
    vectorDB:str
    uid:str

class CsvInput(BaseModel):
    question:str
    file:str
    uid:str

class MongoInput(BaseModel):
    mongo_uri: str
    db_name: str
    question: str

class KnowledgeInput(BaseModel):
    question:str
    uid:str