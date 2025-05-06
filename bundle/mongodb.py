from pymongo import MongoClient
from config import Constant

class MongoDBClient:
    def __init__(self, uri: str=Constant.MONGO_URL, db_name: str='chatbot'):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def get_collection(self, name: str):
        return self.db[name]