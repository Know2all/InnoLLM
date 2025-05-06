from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pymongo import MongoClient
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from config import Constant,logger
from langchain.prompts import PromptTemplate


class MongoActionInput(BaseModel):
    collection: str = Field(..., description="The name of the collection to query.")
    pipeline: List[Dict[str, Any]] = Field(..., description="MongoDB aggregation pipeline.")

class MongoAggregationTool:
    def __init__(self, connection_string: str, db_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.mongo_parser = self._init_mongo_parser()
        self.llm = ChatGoogleGenerativeAI(
            model=Constant.MODEL,
            api_key=Constant.API_KEY,
            temperature=Constant.TEMPERATURE,
            max_tokens=Constant.MAX_TOKENS,
            top_k=Constant.TOP_K,
        )

    def _init_mongo_parser(self):
        return PydanticOutputParser(pydantic_object=MongoActionInput)

    def infer_type_custom(self, value):
        if isinstance(value, str): return "string"
        elif isinstance(value, bool): return "boolean"
        elif isinstance(value, int): return "int"
        elif isinstance(value, float): return "float"
        elif isinstance(value, list):
            if not value: return "array of unknown"
            first = value[0]
            if isinstance(first, dict):
                return [self.infer_schema_custom(first, indent=2)]
            return f"array of {self.infer_type_custom(first)}s"
        elif isinstance(value, dict):
            return self.infer_schema_custom(value, indent=2)
        elif value is None: return "null"
        return type(value).__name__

    def infer_schema_custom(self, doc, indent=0):
        lines, pad = [], " " * indent
        for key, value in doc.items():
            inferred = self.infer_type_custom(value)
            if isinstance(inferred, str):
                lines.append(f'{pad}"{key}": "{inferred}",')
            elif isinstance(inferred, list):
                lines.append(f'{pad}"{key}": [')
                lines.extend(inferred)
                lines.append(f"{pad}],")
            elif isinstance(inferred, dict):
                lines.append(f'{pad}"{key}": {{')
                lines.extend(self.infer_schema_custom(value, indent + 2))
                lines.append(f"{pad}}},")
        return lines

    def get_schema(self, collection_name="customers"):
        try:
            sample = self.db[collection_name].find_one()
            schema_lines = self.infer_schema_custom(sample)
            return "\n".join(schema_lines)
        except Exception as e:
            return f"Schema error: {e}"

    def get_schema_info(self):
        collections = self.db.list_collection_names()
        schema_lines = []
        for name in collections:
            doc = self.db[name].find_one()
            if doc:
                keys = list(doc.keys())
                schema_lines.append(f"{name}: {keys}")
        schema_info = "\n".join(schema_lines)
        logger.info("MongoDB Schema Generated Successfully !")
        return schema_info

    def _build_prompt(self):
        return PromptTemplate.from_template("""
You are a senior MongoDB engineer with deep expertise in writing aggregation pipelines.
You are given a database schema and a natural language question.

Use the following format to respond:

---
Question: <question>
Thought: Think about the best way to query the data based on the schema.
Action: MongoDB Tool
Action Input: {format_instructions}
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question
---

Schema:
{schema}

Begin!
Question: {question}
""")
    
    def _format_result_naturally(self, question: str, raw_result: str) -> str:
        prompt = PromptTemplate.from_template("""
        You are a helpful assistant. You are given a MongoDB aggregation result in JSON format, and a user's original question.

        Your job is to summarize the result in a natural, concise, and human-readable format.

        Original Question:
        {question}

        MongoDB Result:
        {raw_result}

        Answer:
        """)
        
        formatted_prompt = prompt.format(
            question=question,
            raw_result=raw_result
        )
        response = self.llm.invoke(formatted_prompt).content.strip()
        logger.info("Finalizing Natural Response !")
        return response


    def _generate_and_run(self, question: str) -> str:
        # Step 1: Get schema from MongoDB
        schema_info = self.get_schema_info()  # Uses default collection or set manually

        # Step 2: Format prompt
        prompt = self._build_prompt().format(
            question=question,
            schema= schema_info or "No schema available.",
            format_instructions=self.mongo_parser.get_format_instructions()
        )

        # print(prompt)
        logger.info("Mongo Prompt Loaded Successfully !")

        # Step 3: LLM generates action input
        llm_output = self.llm.invoke(prompt).content
        # logger.info(llm_output)
        # print("🔎 Raw LLM Output:\n", llm_output)

        # Step 4: Parse structured pipeline
        try:
            parsed = self.mongo_parser.parse(llm_output)
            logger.info(f"Parsing the MongoTool Response \n {parsed.collection} - {parsed.pipeline} ")
        except Exception as e:
            logger.error(f"Parsing Failed : {e}")
            return f"❌ Parsing failed: {e}"

        # Step 5: Run Mongo pipeline
        result = list(self.db[parsed.collection].aggregate(parsed.pipeline))
        answer = self._format_result_naturally(question, result)
        return answer

    def run(self, query: str) -> str:
        return self._generate_and_run(query)

    async def arun(self, query: str) -> str:
        return self._generate_and_run(query)

if __name__ == "__main__":

    mongoBot = MongoAggregationTool(connection_string=Constant.MONGO_URL,db_name="sample_mflix")
    
    question = "Can you give me top 10 tamil movies do you have ?"

    answer = mongoBot.run(query=question)

    print(answer)