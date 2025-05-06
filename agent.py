import re
from typing import List, Dict, Any, TypedDict
from pydantic import create_model
import requests
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from config import Constant, logger

class APIToolBuilder:
    """A class to build and manage API tools with an LLM workflow and chat history."""

    def __init__(self, routes: List[Dict[str, Any]], model_config: Dict[str, Any]):
        """
        Initialize the APIToolBuilder with API routes, model configuration, and chat history.
        
        Args:
            routes: List of API route definitions
            model_config: Configuration for the LLM model
        """
        self.routes = routes
        self.model_config = model_config
        self.chat_history = ChatMessageHistory()
        self.tools = self._build_tools()
        self.llm = self._initialize_llm()
        self.workflow = self._build_workflow()

    def _sanitize_function_name(self, name: str) -> str:
        """
        Sanitize function names to be valid Python identifiers.
        
        Args:
            name: Raw function name
        Returns:
            Sanitized function name
        """
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        if not name[0].isalpha():
            name = "_" + name
        return name[:64]

    def _map_json_type(self, json_type: str) -> Any:
        """
        Map JSON types to Python types.
        
        Args:
            json_type: JSON type string
        Returns:
            Corresponding Python type
        """
        type_map = {
            "number": int,
            "string": str,
            "boolean": bool,
            "Array Of Objects": List[Dict[str, Any]],
            "array": list,
            "object": dict,
        }
        return type_map.get(json_type, Any)

    def _create_tool_from_api(self, api_def: Dict[str, Any]) -> StructuredTool:
        """
        Create a StructuredTool from an API definition.
        
        Args:
            api_def: API definition dictionary
        Returns:
            Configured StructuredTool
        """
        name = api_def["name"].replace(" ", "_").lower()
        sanitized_name = self._sanitize_function_name(name)
        description = api_def["description"]
        method = api_def.get("method", "GET").upper()
        url = api_def["url"]
        
        body_schema = api_def.get("body", {})
        default_values = api_def.get("default", {})

        InputModel = create_model(
            f"{sanitized_name}_input",
            **{
                key: (
                    self._map_json_type(value),
                    default_values.get(key, ...)
                )
                for key, value in body_schema.items()
            }
        )
        
        def tool_func(**kwargs):
            if method == "POST":
                response = requests.post(url, json=kwargs)
            else:
                response = requests.get(url, params=kwargs)
            return response.json()

        return StructuredTool.from_function(
            name=sanitized_name,
            description=description,
            func=tool_func,
            args_schema=InputModel,
        )

    def _create_decision_tool(self) -> StructuredTool:
        """
        Create a decision tool to control workflow continuation.
        
        Returns:
            Configured StructuredTool for decision making
        """
        DecisionInput = create_model(
            "DecisionInput",
            decision=(str, ...),
            reason=(str, ...)
        )

        def decision_func(decision: str, reason: str) -> Dict[str, str]:
            return {"decision": decision.lower(), "reason": reason}

        return StructuredTool.from_function(
            name="decide_workflow",
            description="Decide whether to continue or end the workflow. Decision must be 'continue' or 'end'.",
            func=decision_func,
            args_schema=DecisionInput,
        )

    def _build_tools(self) -> List[StructuredTool]:
        """
        Build all tools from the provided routes and include the decision tool.
        
        Returns:
            List of StructuredTools
        """
        api_tools = [self._create_tool_from_api(api) for api in self.routes]
        api_tools.append(self._create_decision_tool())
        return api_tools

    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """
        Initialize the LLM with the provided configuration.
        
        Returns:
            Configured ChatGoogleGenerativeAI instance
        """
        return ChatGoogleGenerativeAI(
            model=self.model_config["model"],
            temperature=self.model_config["temperature"],
            api_key=self.model_config["api_key"],
            max_tokens=self.model_config["max_tokens"],
            top_k=self.model_config["top_k"],
        )

    def _should_continue(self, state: MessagesState) -> str:
        """
        Determine if the workflow should continue to tools or end based on messages or decision tool.
        
        Args:
            state: Current MessagesState
        Returns:
            Next node name or END
        """
        messages = state["messages"]
        last_message = messages[-1]

        # Check if the last message contains a decision tool call
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "decide_workflow":
                decision_result = tool_call["args"].get("decision", "continue")
                return "tools" if decision_result == "continue" else END

        # If there are other tool calls, continue to tools
        return "tools" if last_message.tool_calls else END

    def _call_model(self, state: MessagesState) -> Dict[str, List]:
        """
        Call the LLM model with the current state and chat history.
        
        Args:
            state: Current MessagesState
        Returns:
            Updated state with model response
        """
        messages = state["messages"]
        # Combine chat history with current messages
        all_messages = self.chat_history.messages + messages
        response = self.llm.bind_tools(self.tools).invoke(all_messages)
        
        # Update chat history with new messages
        self.chat_history.add_message(messages[-1])
        self.chat_history.add_message(response)
        
        return {"messages": [response]}

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(MessagesState)
        tool_node = ToolNode(tools=self.tools)

        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self._should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def stream(self, input_message: str) -> None:
        """
        Stream the workflow output for a given input message.
        
        Args:
            input_message: Input message to process
        """
        for chunk in self.workflow.stream(
            {"messages": [("human", input_message)]},
            stream_mode="values",
        ):
            chunk["messages"][-1].pretty_print()

    def invoke(self,input:str)->str:
        """
        Generate an answer based on the user query
        Args:
            input: Input from the user
        """
        return self.workflow.invoke({"messages":[('human',input)]})
if __name__ == "__main__":
    # Example usage
    model_config = {
        "model": Constant.MODEL,
        "temperature": Constant.TEMPERATURE,
        "api_key": Constant.API_KEY,
        "max_tokens": Constant.MAX_TOKENS,
        "top_k": Constant.TOP_K,
    }
    
    tool_builder = APIToolBuilder(routes=Constant.ROUTES, model_config=model_config)
    response = tool_builder.stream(f"""
        Tries to provide an answer based on your tools with the following piece of information
        question :Hi
        uid : anwar
        vectorDB: None
    """)