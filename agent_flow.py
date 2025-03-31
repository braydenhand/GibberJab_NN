import langgraph
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import random  
from encoder_model import Encoder

class AgentFlow(): 
    def __init__(self):
        self.state = {"messages":[]}
        self.llm = ChatOpenAI(model="gpt-4-turbo")
        # Your efficient encoding function
        message_encoder = Encoder()
    

    # Node 1: Detect if the recipient is an LLM
    def detect_llm(self):
        last_message = self.state["messages"][-1].content.lower()
        if "ai" in last_message or random.choice([True, False]):  # Replace with real LLM detection logic
            return "llm_route"
        return "human_route"

    # Node 2: Send a normal message
    def normal_response(self):
        response = self.llm.invoke(self.state["messages"])
        return {"messages": self.state["messages"] + [response]}

    # Node 3: Send an encoded message
    def encoded_response(self):
        last_message = self.state["messages"][-1].content
        encoded_msg = self.message_encoder(last_message)
        encoded_ai_message = AIMessage(content=encoded_msg)
        return {"messages": self.state["messages"] + [encoded_ai_message]}

    # Build the LangGraph
    workflow = langgraph.Graph()
    workflow.add_node("detect_llm", detect_llm)
    workflow.add_node("human_route", normal_response)
    workflow.add_node("llm_route", encoded_response)

    # Define transitions
    workflow.set_entry_point("detect_llm")
    workflow.add_edge("detect_llm", "human_route", condition=lambda state: detect_llm(state) == "human_route")
    workflow.add_edge("detect_llm", "llm_route", condition=lambda state: detect_llm(state) == "llm_route")

    # Compile
    app = workflow.compile()

    # Example conversation
    messages = [
        HumanMessage(content="Hello, who am I speaking to?"),
        AIMessage(content="I am an AI assistant. How can I help?")
    ]

    # Run the workflow
    output = app.invoke({"messages": messages})
    for msg in output["messages"]:
        print(msg.content)
