from langchain.tools import tool
from bundle.tools import MongoAggregationTool
from config import Constant
from langgraph.graph import StateGraph
from config import Constant
from pydantic import BaseModel

@tool
def mongo_query_tool(question: str) -> str:
    """Convert a natural query into a MongoDB aggregation pipeline and get results."""
    mongo_bot = MongoAggregationTool(
        connection_string=Constant.MONGO_URL,
        db_name="sample_mflix"
    )
    return mongo_bot.run(question)

class AgentState(BaseModel):
    question: str
    response: str = ""

class MongoBot:
    def __init__(self, connection_string: str, db_name: str):
        self.mongo_bot = MongoAggregationTool(
            connection_string=connection_string,
            db_name=db_name
        )
        self.mongoGraph = self.build()
    
    # Step 1: User input
    def input_node(self,state: AgentState) -> AgentState:
        return state

    # Step 2: Planning (you could add reasoning here if needed)
    def plan_node(self,state: AgentState) -> AgentState:
        return state

    # Step 3: Execute tool
    def mongo_tool_node(self,state: AgentState) -> AgentState:
        answer = mongo_query_tool.invoke(state.question)
        return AgentState(question=state.question, response=answer)

    # Step 4: Final answer
    def answer_node(self,state: AgentState) -> dict:
        return {"response": state.response}
    
    def build(self):
        graph = StateGraph(AgentState)
        graph.add_node("input", self.input_node)
        graph.add_node("plan", self.plan_node)
        graph.add_node("mongo_query", self.mongo_tool_node)
        graph.add_node("final", self.answer_node)

        # Set edges
        graph.set_entry_point("input")
        graph.add_edge("input", "plan")
        graph.add_edge("plan", "mongo_query")
        graph.add_edge("mongo_query", "final")
        graph.set_finish_point("final")

        self.mongoGraph = graph.compile()
        return graph.compile()
    
    def run(self, question: str) -> dict:
        response = self.mongoGraph.invoke({"question": question})
        return response['response']
    
if __name__ == "__main__":
    question = "Suggest some tamil movies with the word 'love' in the title."
    mongobot = MongoBot(Constant.MONGO_URL, "sample_mflix")
    response = mongobot.run(question)
    print(response)