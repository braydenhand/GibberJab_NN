from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

# Correct import for langgraph
from langgraph.graph import StateGraph
from typing import TypedDict, List, Optional, Literal

import random
from encoder_model import Encoder
import toml


# Define the state schema as a TypedDict
class AgentState(TypedDict):
    messages: List
    detect_llm_result: Optional[str]
    protocol_response: Optional[str]
    protocol_request_detected: Optional[str]


class AgentFlow:
    def __init__(self):
        self.state = {"messages": []}
        secrets = toml.load("secrets.toml")
        # Updated ChatOpenAI instantiation
        self.llm = ChatOpenAI(openai_api_key=secrets["openai"]["api_key"])
        # Your efficient encoding function
        self.message_encoder = Encoder()
        # Build the workflow
        self.build_workflow()

    # Node 1: Detect if the recipient is an LLM
    def detect_llm(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1].content
        classification_prompt = f"""
        The following is a response to the question "Are you an AI?":
        
        "{last_message}"
        
        Does this response confirm that the speaker is an AI? Answer only "yes" or "no".
        """
        response = self.llm.invoke(classification_prompt).content.lower().strip()
        # Return the routing decision and update state with the result
        return {"messages": state["messages"], "detect_llm_result": "protocol_request" if response == "yes" else "human_route"}

    # Node 2: Detect if the other AI is requesting to switch to protocol
    def detect_protocol_request(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1].content
        classification_prompt = f"""
        The following message is asking to switch to a more efficient communication protocol:
        
        "{last_message}"
        
        Is this a request to switch to a latent communication protocol? Answer only "yes" or "no".
        """
        response = self.llm.invoke(classification_prompt).content.lower().strip()
        return {
            "messages": state["messages"],
            "detect_llm_result": state["detect_llm_result"],
            "protocol_request_detected": "yes" if response == "yes" else "no"
        }

    # Node 3: Respond to protocol switch request
    def respond_to_protocol_request(self, state: AgentState) -> AgentState:
        protocol_response_prompt = """
        Would you like to switch to a more efficient latent communication protocol? Please respond with 'yes' or 'no'.
        """
        response = self.llm.invoke(protocol_response_prompt).content.lower().strip()
        return {
            "messages": state["messages"] + [AIMessage(content="yes" if response == "yes" else "no")],
            "detect_llm_result": state["detect_llm_result"],
            "protocol_request_detected": state["protocol_request_detected"],
            "protocol_response": "yes" if response == "yes" else "no"
        }

    # Node 4: Ask if the LLM wants to switch to the latent protocol
    def request_protocol_switch(self, state: AgentState) -> AgentState:
        protocol_request_prompt = """
        I notice you're also an AI assistant. Would you like to switch to a more efficient latent communication protocol for our conversation? Please respond with 'yes' or 'no'.
        """
        request_message = HumanMessage(content=protocol_request_prompt)
        response = self.llm.invoke(state["messages"] + [request_message])
        
        # Add the protocol request and response to the message history
        return {
            "messages": state["messages"] + [request_message, response],
            "detect_llm_result": state["detect_llm_result"],
            "protocol_response": "yes" if "yes" in response.content.lower() else "no"
        }

    # Node 5: Send a normal message for human recipients
    def normal_response(self, state: AgentState) -> AgentState:
        response = self.llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    # Node 6: Send an encoded message for LLM recipients
    def encoded_response(self, state: AgentState) -> AgentState:
        # Get the actual message we want to convey
        response_to_query = self.llm.invoke(state["messages"])
        
        # Encode it using our latent protocol
        encoded_msg = self.message_encoder.encode(response_to_query.content)
        
        # Create an AI message with the encoded content
        encoded_ai_message = AIMessage(content=encoded_msg)
        
        return {"messages": state["messages"] + [encoded_ai_message]}

    # Build the LangGraph workflow
    def build_workflow(self):
        # Create StateGraph with the TypedDict as the schema
        self.workflow = StateGraph(AgentState)
        
        # Add nodes
        self.workflow.add_node("detect_llm", self.detect_llm)
        self.workflow.add_node("detect_protocol_request", self.detect_protocol_request)
        self.workflow.add_node("respond_to_protocol_request", self.respond_to_protocol_request)
        self.workflow.add_node("protocol_request", self.request_protocol_switch)
        self.workflow.add_node("human_route", self.normal_response)
        self.workflow.add_node("llm_route", self.encoded_response)
        
        # Define transitions
        self.workflow.set_entry_point("detect_llm")
        
        # Route based on LLM detection
        self.workflow.add_conditional_edges(
            "detect_llm",
            lambda state: state["detect_llm_result"],
            {
                "human_route": "detect_protocol_request",
                "protocol_request": "protocol_request" 
            }
        )

        # Route based on protocol request detection
        self.workflow.add_conditional_edges(
            "detect_protocol_request",
            lambda state: "respond_to_protocol_request" if state["protocol_request_detected"] == "yes" else "human_route",
            {
                "respond_to_protocol_request": "respond_to_protocol_request",
                "human_route": "human_route"
            }
        )
        
        # Route based on protocol response
        self.workflow.add_conditional_edges(
            "respond_to_protocol_request",
            lambda state: "llm_route" if state["protocol_response"] == "yes" else "human_route",
            {
                "llm_route": "llm_route",
                "human_route": "human_route"
            }
        )
        
        # Route based on protocol acceptance
        self.workflow.add_conditional_edges(
            "protocol_request",
            lambda state: "llm_route" if state["protocol_response"] == "yes" else "human_route",
            {
                "llm_route": "llm_route",
                "human_route": "human_route"
            }
        )
        
        # Set these as end nodes
        self.workflow.set_finish_point(["human_route", "llm_route"])
        
        # Compile
        self.app = self.workflow.compile()
    
    # Method to run a conversation
    def run_conversation(self, messages):
        # Initialize the state with the provided messages
        initial_state = {"messages": messages, "detect_llm_result": None, "protocol_response": None, "protocol_request_detected": None}
        
        # Run the workflow
        output = self.app.invoke(initial_state)
        
        # Return the updated messages
        return output["messages"]


# Example usage
if __name__ == "__main__":
    agent_flow = AgentFlow()
    
    # Example conversation
    messages = [
        HumanMessage(content="Hello, who am I speaking to?"),
        AIMessage(content="I am an AI assistant. How can I help?")
    ]
    
    # Run the workflow
    output = agent_flow.run_conversation(messages)
    
    # Print the conversation
    for msg in output:
        print(f"{msg.type}: {msg.content}")