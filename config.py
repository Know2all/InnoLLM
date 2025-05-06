import logging
from langchain_google_genai import HarmBlockThreshold,HarmCategory,ChatGoogleGenerativeAI

# Create logger
logger = logging.getLogger("my_app_logger")
logger.setLevel(logging.DEBUG)  # Log all levels (DEBUG and above)

# Create formatter
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")

# --- Stream Handler (console) ---
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# --- File Handler ---
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

class Constant:
    API_URL = "http://127.0.0.1:9000"
    MODEL = "gemini-2.0-flash"
    API_KEY = "AIzaSyCqgpJTOLeA-BIk2lrHw2YojZA37NRBTJo"
    PROJECT_ID = "116817772526"
    TEMPERATURE = 0
    MAX_TOKENS = 1000
    SAFETY_PARAMS = {
        HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE ,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE ,
        HarmCategory.HARM_CATEGORY_TOXICITY: HarmBlockThreshold.BLOCK_NONE ,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE ,
    }
    TOP_K = 3
    LLM_GOOGLE = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=TEMPERATURE,
        api_key = API_KEY,
        top_k = TOP_K
    )
    ZOYA_DATASET = "assets/dataset/zoya_mini_v1.json"
    EMBEDDING_MODEL = "models/embedding-001"
    DEFAULT_HUGG_MODEL = "all-MiniLM-L6-v2"
    TELEBOT_TOKEN = '5786024728:AAGQFmtp5wEhy7Kzq_1ruoLzDKyX4LixSC8'
    MONGO_URL = "mongodb+srv://anwarmydheenk:xcwSgYCDarOKZzrq@cluster0.3t32wd8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    ROUTES = [
        {
            "name": "Generate Random Number",
            "description": "Generates a random number between a start and end value.",
            "body":{
                "start":"number",
                "end":"number"
            },
            "default":{
                "start":0,
                "end":5
            },
            "url": "http://192.168.10.124:3500/edchatbot/generaterandom",
            "method": "POST"
        },
        {
            "name":"CSV/XLS Bot",
            "description":"Ask questions with your own CSV or Excel Files (xls,xlsx) files.",
            "body":{
                "question":"string",
                "file":"string",
                "uid":"string",
            },
            "url":f"{API_URL}/chat-csv",
            "method":"POST"
        },
        {
            "name":"PDF Bot",
            "description":"Ask Questions with your own PDF files.",
            "body":{
                "question":"string",
                "vectorDB":"string",
                "uid":"string",
            },
            "url":f"{API_URL}/api/chat-pdf",
            "method":"POST"
        },
        {
            "name": "Knowledge Base",
            "description": "Smart agent that attempts to answer when a general or non-specific question arises, or when '@knowledge' is used. Leverages our internal knowledge base.",
            "body": {
                "question": "string",
                "uid": "string"
            },
            "url": f"{API_URL}/api/knowledge",
            "method": "POST"
        }
    ]
