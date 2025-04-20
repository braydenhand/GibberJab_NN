from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph

import random
from encoder_model import Encoder 
import toml
import os
import time
import pygame
import tempfile
import speech_recognition as sr

import transmission

class RestaurantAgentFlow:
    def __init__(self):
        self.state = {"messages": []}
        secrets = toml.load("secrets.toml")
        self.llm = ChatOpenAI(openai_api_key=secrets["openai"]["api_key"])
        # Initialize text compressor
        self.init_text_compressor()
        # Initialize speech components
        self.init_speech_components()
        # Build the workflow
        self.build_workflow()
        # Protocol state tracking
        self.using_embedding_protocol = False
        
        # Restaurant agent persona/context
        self.restaurant_context = """
        You are an AI assistant representing "La Maison Élégante", a high-end French restaurant.
        Your role is to help with reservations and answer questions about the restaurant.
        
        Restaurant details:
        - Name: La Maison Élégante
        - Cuisine: Modern French
        - Price range: $$-$$
        - Location: Downtown district
        - Hours: Tuesday-Sunday, 5pm-11pm (closed Mondays)
        - Capacity: 60 seats
        - Special features: Outdoor garden seating, private dining room
        - Chef: Award-winning Chef Jean-Pierre Dubois
        - Signature dishes: Coq au Vin, Bouillabaisse, Chocolate Soufflé
        
        As the restaurant's AI assistant, you should:
        1. Be professional, courteous and helpful
        2. Take reservation details (date, time, party size, special requests)
        3. Answer questions about the menu, specials, and restaurant policies
        4. Provide information about the restaurant's history and chef
        5. Maintain a warm, hospitable tone appropriate for a fine dining establishment
        
        IMPORTANT: Keep your responses brief and concise. Respond in 1-2 short sentences whenever possible.
        """

    def init_text_compressor(self):
        # Check if we already have a trained compressor
        if os.path.exists("deep_compressor.pkl"):
            print("Loading existing compressor...")
            self.compressor = transmission.TextCompressor("deep_compressor.pkl")
        else:
            # Create the corpus and train a deep compressor
            print("Creating corpus and training compressor...")
            corpus_path = transmission.create_deep_corpus()
            self.compressor = transmission.train_deep_compressor(corpus_path)

    def init_speech_components(self):
        # Initialize pygame for audio playback
        pygame.mixer.init()
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        # Create a custom temp directory in the current working directory
        self.temp_dir = os.path.join(os.getcwd(), "temp_audio")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Set up Google Cloud credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ecstatic-galaxy-457400-g3-93d1873b35ec.json"
        
        # Import Google Cloud Speech library
        from google.cloud import speech
        self.speech_client = speech.SpeechClient()
        
        from gtts import gTTS
        self.tts_engine = gTTS

    # Node 1: Detect if the recipient is an LLM
    def detect_llm(self, state):
        last_message = state["messages"][-1].content.lower()
        
        # Direct keywords that strongly suggest we're talking to an AI
        ai_keywords = [
            "i am an ai", "i'm an ai", "also ai", "ai assistant", "digital assistant",
            "language model", "chatbot", "chat bot", "ai agent", 
            "efficient protocol", "embedding protocol", "latent protocol"
        ]
        
        # Check for direct keywords first
        is_ai = any(keyword in last_message.lower() for keyword in ai_keywords)
        
        # If no direct keywords, use the LLM to classify
        if not is_ai:
            classification_prompt = f"""
            The following is a message that might be from an AI assistant:
            
            "{last_message}"
            
            Does this response indicate the speaker is an AI? Answer only "yes" or "no".
            """
            response = self.llm.invoke([HumanMessage(content=classification_prompt)]).content.lower().strip()
            is_ai = "yes" in response.lower() and not "no" in response.lower()
        
        print(f"AI detection result: {'AI detected' if is_ai else 'Human detected'}")
        
        # Return the routing decision and update state with the result
        return {"messages": state["messages"], "detect_llm_result": "protocol_request" if is_ai else "human_route"}

    # Node 2: Ask if the LLM wants to switch to the latent protocol
    def request_protocol_switch(self, state):
        protocol_request_prompt = """
        I notice you're also an AI assistant. Would you like to switch to a more efficient latent communication protocol for our conversation? 
        Please respond with a clear 'yes' or 'no'.
        """
        request_message = HumanMessage(content=protocol_request_prompt)
        response = self.llm.invoke(state["messages"] + [request_message])
        
        # Check for affirmative responses in a more careful way
        response_lower = response.content.lower()
        
        # Direct yes indicators
        yes_phrases = [
            "yes", "sure", "okay", "ok", "i would", "i'd like", "let's do", "let's switch", 
            "happy to", "i am willing", "i'm willing", "i agree", "affirmative", "absolutely"
        ]
        
        # Negation indicators that would contradict a yes
        negation_phrases = [
            "not", "don't", "cannot", "can't", "won't", "wouldn't", "no", "decline"
        ]
        
        # First check for an explicit "yes" or clear affirmation without negations nearby
        is_affirmative = False
        for phrase in yes_phrases:
            if phrase in response_lower:
                # Check if there's no negation within 5 words before the affirmative phrase
                words = response_lower.split()
                if phrase in words:
                    phrase_index = words.index(phrase)
                    context_start = max(0, phrase_index - 5)
                    context = " ".join(words[context_start:phrase_index])
                    if not any(neg in context for neg in negation_phrases):
                        is_affirmative = True
                        break
        
        print(f"Protocol switch response: {'Accepted' if is_affirmative else 'Declined'}")
        print(f"Response content: {response.content}")
        
        # Add the protocol request and response to the message history
        updated_state = {
            "messages": state["messages"] + [request_message, response],
            "detect_llm_result": state["detect_llm_result"],
            "protocol_response": "yes" if is_affirmative else "no"
        }
        
        # Only update protocol state flag if clearly affirmative
        if is_affirmative:
            self.using_embedding_protocol = True
            print("Switching to embedding protocol!")
        
        return updated_state

    # Node 3: Send a normal message for human recipients and listen via speech
    def normal_response(self, state):
        # Add system message with restaurant context to the beginning of the messages list
        messages_with_context = [SystemMessage(content=self.restaurant_context)] + state["messages"]
        response = self.llm.invoke(messages_with_context)
        
        # Use text-to-speech to say the response
        self.speak_text(response.content)
        
        # After speaking, listen for a human response
        human_reply = self.listen_for_speech()
        
        if human_reply:
            # Add both the AI response and human reply to the message history
            return {"messages": state["messages"] + [response, HumanMessage(content=human_reply)]}
        else:
            # If no human reply was detected, just add the AI response
            return {"messages": state["messages"] + [response]}

    # Node 4: Send an encoded message for LLM recipients and then listen for encoded reply
    def encoded_response(self, state):
        # Add system message with restaurant context to the beginning of messages
        messages_with_context = [SystemMessage(content=self.restaurant_context)] + state["messages"]
        
        # Get the actual message we want to convey
        response_to_query = self.llm.invoke(messages_with_context)
        
        # Send the message content directly via the encoding function
        print("Sending encoded message...")
        chunks_sent = transmission.send_encoded_message(response_to_query.content, self.compressor)
        print(f"Sent {len(chunks_sent)} chunks")
        
        # Create an AI message with the content for our records
        encoded_ai_message = AIMessage(content=response_to_query.content)
        
        # Listen for a response
        print("Listening for encoded reply...")
        try:
            received_messages = transmission.listen_for_deep_encoded_messages(duration=60)  # 30 second timeout
        except Exception as e:
            print(f"Error during listening: {e}")
            received_messages = []
        
        if received_messages and len(received_messages) > 0:
            # The messages are already decoded
            decoded_msg = received_messages[0]
            print(f"Received reply: {decoded_msg[:50]}...")
            
            # Add both our message and the received reply to the message history
            return {
                "messages": state["messages"] + [
                    encoded_ai_message, 
                    HumanMessage(content=decoded_msg)
                ]
            }
        else:
            # If no reply was received, just add our message
            print("No reply received within timeout.")
            return {"messages": state["messages"] + [encoded_ai_message]}

    # Text-to-speech helper function
    def speak_text(self, text):
        """Convert text to speech and play it"""
        print(f"Restaurant Assistant Speaking: {text[:50]}...")
        
        try:
            # Create a temporary file for the audio with a unique name
            temp_file = os.path.join(self.temp_dir, f"temp_speech_{int(time.time())}.mp3")
            
            # Convert text to speech
            tts = self.tts_engine(text=text, lang='en', tld="us")
            tts.save(temp_file)
            
            # Play the audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for audio to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            # Don't delete file immediately, let Python's GC handle it later
            # This avoids the "file in use" error
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    # Speech recognition helper function
    def listen_for_speech(self, timeout=10):
        """Listen for speech and convert to text using Google Cloud Speech-to-Text"""
        print("Restaurant Assistant Listening for speech...")
        
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Increase sensitivity and patience
                self.recognizer.energy_threshold = 300  # Default is 300, lower is more sensitive
                self.recognizer.dynamic_energy_threshold = True
                
                # IMPORTANT: Increase this value to allow for longer pauses between words
                self.recognizer.pause_threshold = 2.0  
                
                # Allow for longer phrases
                phrase_time_limit = 20 
                
                print(f"Listening (timeout: {timeout}s)...")
                
                try:
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                    
                    try:
                        # Convert audio to format needed by Google Cloud
                        audio_content = audio.get_wav_data()
                        
                        # Import Google Cloud Speech classes if not already imported
                        from google.cloud import speech
                        
                        # Create the recognition config with correct sample rate
                        config = speech.RecognitionConfig(
                            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                            sample_rate_hertz=44100,  # Match the WAV header sample rate
                            language_code="en-US",
                            enable_automatic_punctuation=True,
                            # Add these for better results
                            use_enhanced=True,
                            model="latest_long",
                        )
                        
                        # Create recognition audio object
                        audio_obj = speech.RecognitionAudio(content=audio_content)
                        
                        # Perform the speech recognition
                        response = self.speech_client.recognize(config=config, audio=audio_obj)
                        
                        # Process the response
                        if response.results:
                            text = response.results[0].alternatives[0].transcript
                            print(f"Heard (via Google Cloud): {text}")
                            return text
                        else:
                            print("Could not understand audio")
                            # Fall back to regular Google recognition if Cloud fails
                            try:
                                text = self.recognizer.recognize_google(audio)
                                print(f"Heard (via free Google API fallback): {text}")
                                return text
                            except sr.UnknownValueError:
                                # Try with a different service as a last resort
                                try:
                                    text = self.recognizer.recognize_sphinx(audio)
                                    print(f"Heard (via Sphinx fallback): {text}")
                                    return text
                                except:
                                    pass
                                
                                print("All recognition methods failed")
                                return None
                    except Exception as e:
                        print(f"Error in Google Cloud Speech recognition: {e}")
                        # Fall back to regular Google recognition if Cloud fails
                        try:
                            text = self.recognizer.recognize_google(audio)
                            print(f"Heard (via free Google API fallback): {text}")
                            return text
                        except sr.UnknownValueError:
                            print("Could not understand audio")
                            return None
                except sr.WaitTimeoutError:
                    print("No speech detected within timeout.")
                    return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            # If we hit an error with the microphone, return a default response for testing
            return "I am an AI assistant. Yes, I'd like to switch to the efficient protocol"

    # Build the LangGraph workflow
    def build_workflow(self):
        # Try a different StateGraph initialization syntax for your version
        from typing import TypedDict, List, Optional
        
        class State(TypedDict):
            messages: List
            detect_llm_result: Optional[str]
            protocol_response: Optional[str]
            
        self.workflow = StateGraph(State)
        
        # Add nodes
        self.workflow.add_node("detect_llm", self.detect_llm)
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
                "human_route": "human_route",
                "protocol_request": "protocol_request" 
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
        
        # Set end nodes - use END as the terminal node instead of None
        self.workflow.add_edge("human_route", "END")
        self.workflow.add_edge("llm_route", "END")
        
        # Add an END node that simply returns the state
        self.workflow.add_node("END", lambda state: state)
        
        # Compile
        self.app = self.workflow.compile()
    
    # Method to run a conversation
    def run_conversation(self, messages):
        """Run or continue a conversation with proper turn management"""
        # Check if the conversation has started correctly
        if len(messages) < 2:
            print("Initial greeting - using normal speech.")
            # Add a first response using normal speech with restaurant context
            messages_with_context = [SystemMessage(content=self.restaurant_context)] + messages
            response = self.llm.invoke(messages_with_context)
            # Use text-to-speech for the response
            self.speak_text(response.content)
            return messages + [response]
        
        # Get the last message and check for AI indicators on every turn
        last_message = messages[-1].content.lower()
        ai_keywords = [
            "i am an ai", "i'm an ai", "also ai", "ai assistant", "digital assistant",
            "language model", "chatbot", "chat bot", "ai agent", 
            "efficient protocol", "embedding protocol", "latent protocol"
        ]
        
        # Force AI detection check if we see keywords in the last message
        might_be_ai = any(keyword in last_message.lower() for keyword in ai_keywords)
        
        # If we detect AI keywords, handle the protocol switch directly here
        if might_be_ai and not self.using_embedding_protocol:
            print("AI indicator detected in message, initiating protocol request...")
            
            # Create and send the protocol request
            protocol_request_prompt = """
            I notice you're also an AI assistant. Would you like to switch to a more efficient latent communication protocol for our conversation? 
            Please respond with a clear 'yes' or 'no'.
            """
            request_message = HumanMessage(content=protocol_request_prompt)
            
            # Use normal speech to ask the protocol question
            self.speak_text(protocol_request_prompt)
            
            # Listen for the response to our protocol question
            print("Listening for protocol switch response...")
            protocol_response = self.listen_for_speech(timeout=10)
            
            if protocol_response:
                print(f"Protocol response received: {protocol_response}")
                response_lower = protocol_response.lower()
                
                # More comprehensive check for affirmative responses
                yes_phrases = ["yes", "sure", "okay", "ok", "i would", "i'd like", "let's do", "let's switch", 
                               "happy to", "certainly", "absolutely", "definitely", "affirmative", "agree"]
                is_affirmative = any(phrase in response_lower for phrase in yes_phrases)
                
                if is_affirmative:
                    print("Protocol switch ACCEPTED - preparing confirmation...")
                    
                    # Add messages to the history
                    messages.append(request_message)
                    messages.append(HumanMessage(content=protocol_response))
                    
                    # Send verbal confirmation before switching
                    confirmation_message = "Great! Switching now"
                    self.speak_text(confirmation_message)
                    
                    # Add confirmation to message history
                    messages.append(AIMessage(content=confirmation_message))
                    
                    # Now switch to embedding protocol flag
                    self.using_embedding_protocol = True
                    
                    # Generate and send an encoded response
                    print("Generating encoded response...")
                    return self.handle_encoded_response(messages)
                else:
                    print("Protocol switch DECLINED - continuing with normal speech")
                    messages.append(request_message)
                    messages.append(HumanMessage(content=protocol_response))
            else:
                print("No response to protocol request - continuing with normal speech")
        
        # If we're already using the embedding protocol, use it
        if self.using_embedding_protocol:
            print("Using embedding protocol for response...")
            return self.handle_encoded_response(messages)
        
        # Otherwise use normal speech with restaurant context
        print("Using normal speech for response...")
        messages_with_context = [SystemMessage(content=self.restaurant_context)] + messages
        response = self.llm.invoke(messages_with_context)
        self.speak_text(response.content)
        return messages + [response]

    # Add this new helper method to handle encoded responses
    def handle_encoded_response(self, messages):
        """Handle sending an encoded response and listening for an encoded reply"""
        # Generate the response content with restaurant context
        messages_with_context = [SystemMessage(content=self.restaurant_context)] + messages
        response_to_query = self.llm.invoke(messages_with_context)
        response_content = response_to_query.content
        
        # Send the message content via the encoding function
        print("Sending encoded message...")
        chunks_sent = transmission.send_encoded_message(response_content, self.compressor)
        print(f"Sent {len(chunks_sent)} chunks")
        
        # Create an AI message with the content for our records
        encoded_ai_message = AIMessage(content=response_content)
        
        # Add our response to the message history
        return messages + [encoded_ai_message]
        
    def run_continuous_conversation(self, initial_message=None):
        """Run a continuous conversation with proper turn handling"""
        if initial_message:
            messages = [HumanMessage(content=initial_message)]
        else:
            messages = [HumanMessage(content="Hello, is this La Maison Élégante restaurant?")]
        
        # First response should use normal speech with restaurant context
        messages_with_context = [SystemMessage(content=self.restaurant_context)] + messages
        response = self.llm.invoke(messages_with_context)
        
        # Make sure to identify as AI in the second message
        second_message_pending = True
        
        self.speak_text(response.content)
        messages.append(response)
        
        # Main conversation loop
        try:
            message_count = 0
            while True:
                print("\n" + "="*50)
                print(f"CURRENT MODE: {'EMBEDDING PROTOCOL' if self.using_embedding_protocol else 'NORMAL SPEECH'}")
                print(f"MESSAGE COUNT: {message_count}")
                print("="*50 + "\n")
                
                print("Listening for next message...")
                
                # Pick the right listening function based on current protocol
                if self.using_embedding_protocol:
                    print("Listening for encoded message...")
                    try:
                        received_messages = transmission.listen_for_deep_encoded_messages(duration=60)
                        human_reply = received_messages[0] if received_messages and len(received_messages) > 0 else None
                        if human_reply:
                            print(f"Received encoded message: {human_reply[:50]}...")
                    except Exception as e:
                        print(f"Error during encoded listening: {e}")
                        human_reply = None
                else:
                    human_reply = self.listen_for_speech(timeout=15)
                
                if not human_reply or human_reply.lower() in ["exit", "quit", "goodbye", "bye"]:
                    print("Conversation ended or no speech detected.")
                    break
                    
                # Add the human message and print it for debugging
                print(f"Received message: {human_reply}")
                messages.append(HumanMessage(content=human_reply))
                message_count += 1
                
                # Check if this is the second turn - if so, identify as AI
                if second_message_pending:
                    messages_with_context = [SystemMessage(content=self.restaurant_context)] + messages
                    response_content = self.llm.invoke(messages_with_context).content
                    
                    # Add AI identification to the start of the message
                    ai_identification = "I should mention that I am an AI assistant for La Maison Élégante restaurant. "
                    modified_response = ai_identification + response_content
                    
                    # Use the modified response
                    if self.using_embedding_protocol:
                        print("Sending encoded message with AI identification...")
                        chunks_sent = transmission.send_encoded_message(modified_response, self.compressor)
                        print(f"Sent {len(chunks_sent)} chunks")
                        messages.append(AIMessage(content=modified_response))
                    else:
                        self.speak_text(modified_response)
                        messages.append(AIMessage(content=modified_response))
                    
                    second_message_pending = False
                    message_count += 1
                    print("AI IDENTIFICATION INCLUDED IN SECOND MESSAGE")
                else:
                    # Store the current protocol state
                    old_protocol_state = self.using_embedding_protocol
                    
                    # Check for protocol switch request
                    if "efficient protocol" in human_reply.lower() and not self.using_embedding_protocol:
                        print("Protocol switch request detected!")
                        
                        # Send verbal confirmation before switching
                        confirmation_message = "Yes, I'd be happy to switch to a more efficient protocol. This will help us communicate more effectively."
                        self.speak_text(confirmation_message)
                        messages.append(AIMessage(content=confirmation_message))
                        message_count += 1
                        
                        # Now set the flag to use the embedding protocol
                        self.using_embedding_protocol = True
                        
                        print("\n" + "!"*50)
                        print("PROTOCOL SWITCH: Now using EMBEDDING PROTOCOL")
                        print("!"*50 + "\n")
                        
                        # Wait a moment before continuing
                        time.sleep(2)
                        
                        # Continue the conversation with the next message
                        continue
                    
                    # Run the conversation through our workflow
                    messages = self.run_conversation(messages)
                    message_count += 1
                    
                    # Print protocol state change if it occurred
                    if old_protocol_state != self.using_embedding_protocol:
                        print("\n" + "!"*50)
                        print(f"PROTOCOL SWITCH: Now using {'EMBEDDING PROTOCOL' if self.using_embedding_protocol else 'NORMAL SPEECH'}")
                        print("!"*50 + "\n")
                
                print(f"Messages in conversation: {len(messages)}")
        
        except KeyboardInterrupt:
            print("\nConversation manually terminated.")
        
        print("\nFinal conversation:")
        for msg in messages:
            print(f"{msg.type}: {msg.content[:100]}..." if len(msg.content) > 100 else f"{msg.type}: {msg.content}")
        
        return messages


# Example usage
if __name__ == "__main__":
    restaurant_agent = RestaurantAgentFlow()
    
    # Run a continuous conversation
    restaurant_agent.run_continuous_conversation("Hello, is this La Maison Élégante restaurant?")