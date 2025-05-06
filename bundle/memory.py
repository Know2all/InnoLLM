from bundle.mongodb import MongoDBClient
from datetime import datetime

class ConversationMemory:
    def __init__(self, db_client: MongoDBClient, collection_name: str):
        self.collection = db_client.get_collection(collection_name)

    def save_conversation(self, question: str, answer: str, uid: str) -> str:
        conversation = {
            "uid": uid,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now()
        }
        result = self.collection.insert_one(conversation)
        return str(result.inserted_id)

    def get_conversations(self, uid: str, limit: int = 5):
        return list(self.collection.find({"uid": uid}).sort("timestamp", -1).limit(limit))