import os
from config import Constant,logger
import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from bundle.mongodb import MongoDBClient
from bundle.memory import ConversationMemory

class PandasAgent:
    def __init__(self,csv_file,uid,conversation_memory:ConversationMemory,llm=ChatGoogleGenerativeAI(model=Constant.MODEL,temperature=Constant.TEMPERATURE,api_key=Constant.API_KEY)):
        logger.info("Pandas Tool V2 Initialized")
        self.llm = llm
        self.uid = uid
        self.conversation_memory = conversation_memory
        self.csv_file = csv_file
        self.df = self.load_dataframe(csv_file)
        self.tools = self.build_tools()
        self.sys_msg = SystemMessage(content=f"""
You are a smart assistant that reasons step-by-step to solve questions about a pandas DataFrame called `df`.

### Important Guidelines:
- If the question involves data exploration (like number of rows, column names, data types, value counts, missing values, etc.), ALWAYS use the `python_repl_ast` tool to inspect the DataFrame.
- NEVER guess or assume — use code to verify the answer whenever possible.
- Treat string comparisons in a **case-insensitive** way (`.str.lower()`, `.str.contains(..., case=False)`).
- Always safely handle null values (`.fillna()`, `.dropna()`, `.isnull()`, `.notnull()`).

You have access to a Python code executor tool:
- **`python_repl_ast`**: Executes Python code on the DataFrame `df`. Use it whenever you need to analyze, filter, or summarize the data.

The DataFrame has the following columns:
{self.df.columns.tolist()}
""")

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.react_graph = self.build_react_graph()
        self.messages = []
        self.load_chat_memory()

    
    def load_chat_memory(self):
        memories = self.conversation_memory.get_conversations(uid=self.uid)
        for conversation in memories:
            self.messages.append(HumanMessage(content=conversation['question']))
            self.messages.append(AIMessage(content=conversation['answer']))

    def load_dataframe(self, path)->pd.DataFrame:
        logger.info(f"Loading DataFrame from {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path, encoding="ISO-8859-1")
        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def build_tools(self):
        python_ast_repl = PythonAstREPLTool(globals={'df':self.df})
        python_ast_repl.description="Executes Python code on a pandas DataFrame `df`. Input should be valid Python using the `df` variable."
        return [python_ast_repl]
    
    def reasoner(self,state: MessagesState):
        return {"messages": [self.llm_with_tools.invoke([self.sys_msg] + state["messages"])]}
    
    def build_react_graph(self):
        builder = StateGraph(MessagesState)

        builder.add_node("reasoner", self.reasoner)
        builder.add_node("tools", ToolNode(self.tools)) # for the tools


        builder.add_edge(START, "reasoner")
        builder.add_conditional_edges(
            "reasoner",
            # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
            # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        builder.add_edge("tools", "reasoner")
        react_graph = builder.compile()
        return react_graph
    
    def run(self,question):
        self.messages = [self.sys_msg] + self.messages + [HumanMessage(content=question)]
        result = self.react_graph.invoke({"messages": self.messages})
        answer = result['messages'][-1].content
        if answer.strip() == "":
            answer = "I don't know the answer to that question."
        self.conversation_memory.save_conversation(question=question,answer=answer,uid=self.uid)
        return answer