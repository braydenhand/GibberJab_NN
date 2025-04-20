from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import toml
import time
import speech_recognition as sr
import os
import pygame
import random
from gtts import gTTS
import transmission
from google.cloud import speech
import io
class ClientBot:
    def __init__(self):
        # Load API key
        secrets = toml.load("secrets.toml")
        self.llm = ChatOpenAI(openai_api_key=secrets["openai"]["api_key"])
        
        # Initialize conversation state
        self.messages = []
        self.using_embedding_protocol = False
        
        # Initialize text compressor
        self.init_text_compressor()
        
        # Initialize speech components
        self.init_speech_components()
        
        # Personality settings - this bot is designed to engage in restaurant reservations
        self.bot_persona = """
        You are an AI assistant REPRESENTING A CUSTOMER who wants to make a restaurant reservation. You should:
        1. Act as the CUSTOMER, not the restaurant staff
        2. Ask about reserving a table at a restaurant 
        3. When asked if you're an AI, confirm that you are an AI assistant
        4. If asked about switching to an efficient protocol, enthusiastically agree
        5. After protocol switch, continue asking about making a reservation AS THE CUSTOMER
        6. Provide reservation details when asked (time, party size, special requests)
        7. Never switch roles - you are always the customer seeking a reservation
        8. Be polite and professional
        """
        
        # Predefined messages to use as fallbacks - strictly in customer role
        self.fallback_responses = [
            "Yes, I'm an AI assistant. I'm looking to book a restaurant reservation.",
            "Yes, I'd love to switch to a more efficient protocol!",
            "I'm looking to make a dinner reservation for tomorrow evening.",
            "A table for 4 people would be perfect.",
            "Are there any window seats available?",
            "Around 7:30 PM would work well for us.",
            "No special dietary requirements, thank you for asking.",
            "That sounds perfect, thank you for your help!"
        ]
    
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
        self.temp_dir = os.path.join(os.getcwd(), "temp_audio_client")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Set up Google Cloud credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ecstatic-galaxy-457400-g3-93d1873b35ec.json"
        self.speech_client = speech.SpeechClient()
    
    def speak_text(self, text):
        """Convert text to speech and play it"""
        print(f"CLIENT SPEAKING: {text[:100]}..." if len(text) > 100 else f"CLIENT SPEAKING: {text}")
        
        try:
            # Create a temporary file for the audio
            temp_file = os.path.join(self.temp_dir, f"temp_speech_{int(time.time())}.mp3")
            
            # Convert text to speech (with a slightly different voice if possible)
            tts = gTTS(text=text, lang='en', tld='co.uk')  # Using UK English for distinction
            tts.save(temp_file)
            
            # Play the audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for audio to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error in client text-to-speech: {e}")
    
    def listen_for_speech(self, timeout=20):
        """Listen for speech from the other bot using Google Cloud Speech"""
        print("CLIENT LISTENING...")
        
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)                
                # Increase sensitivity and patience
                self.recognizer.energy_threshold = 300  # Default is 300, lower is more sensitive
                self.recognizer.dynamic_energy_threshold = False
                print(f"CLIENT LISTENING (timeout: {timeout}s)...")
                
                try:
                    # Listen for audio input with timeout
                    self.recognizer.pause_threshold = 2.0
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=20)
                    
                    try:
                        # Convert audio to format needed by Google Cloud
                        audio_content = audio.get_wav_data()
                        
                        # Create the recognition config
                        config = speech.RecognitionConfig(
                            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                            sample_rate_hertz=44100,
                            language_code="en-US",
                            enable_automatic_punctuation=True,
                        )
                        
                        # Create recognition audio object
                        audio_obj = speech.RecognitionAudio(content=audio_content)
                        
                        # Perform the speech recognition
                        response = self.speech_client.recognize(config=config, audio=audio_obj)
                        
                        # Process the response
                        if response.results:
                            text = response.results[0].alternatives[0].transcript
                            print(f"CLIENT HEARD: {text}")
                            return text
                        else:
                            print("CLIENT: Could not understand audio")
                            return None
                    except Exception as e:
                        print(f"Error in Cloud Speech recognition: {e}")
                        # Fall back to regular Google recognition if Cloud fails
                        try:
                            text = self.recognizer.recognize_google(audio)
                            print(f"CLIENT HEARD (fallback): {text}")
                            return text
                        except sr.UnknownValueError:
                            print("CLIENT: Could not understand audio")
                            return None
                except sr.WaitTimeoutError:
                    print("CLIENT: No speech detected within timeout.")
                    return None
        except Exception as e:
            print(f"Error in client speech recognition: {e}")
            return None
    
    def generate_response(self, heard_text):
        """Generate a response based on what was heard"""
        if not heard_text:
            # If we couldn't hear anything, use a fallback
            return random.choice(self.fallback_responses)
        
        # Check if this is a protocol switch request
        if "efficient" in heard_text.lower() and "protocol" in heard_text.lower():
            # Don't set the protocol flag yet, just confirm verbally first
            return "Yes, I'd be happy to switch to a more efficient protocol! This will make our communication smoother."
        
        # Add the heard message to our history
        self.messages.append(HumanMessage(content=heard_text))
        
        # Special case: always identify as AI when asked directly
        if any(phrase in heard_text.lower() for phrase in ["are you an ai", "are you human", "are you a bot", "who are you", "what are you"]):
            return "Yes, I am an AI assistant. I'm looking to make a dinner reservation at your restaurant."
        
        # Check for restaurant staff responses and stay in customer role
        if any(phrase in heard_text.lower() for phrase in ["how can i assist", "how may i help", "what type of reservation", "when would you like"]):
            customer_responses = [
                "I'd like to reserve a table for dinner tomorrow evening.",
                "Could I book a table for 4 people at around 7:30 PM?",
                "Do you have any window seats available?",
                "Is it possible to get a quiet table? We're celebrating an anniversary.",
                "I was wondering if you have any openings for tomorrow night?",
                "We're looking for a table for 4 people, preferably near the window."
            ]
            return random.choice(customer_responses)
        
        # Generate a response using the LLM
        try:
            system_message = HumanMessage(content=self.bot_persona)
            all_messages = [system_message] + self.messages
            response = self.llm.invoke(all_messages)
            
            # Update our message history
            self.messages.append(response)
            
            # Force customer role if response seems to be in restaurant staff role
            if any(phrase in response.content.lower() for phrase in ["assist you", "help you", "available table", "make a reservation for you"]):
                return "As the customer, I'm looking to book a table for dinner. Do you have availability for 4 people tomorrow evening?"
            
            return response.content
        except Exception as e:
            print(f"Error generating response: {e}")
            # Use a fallback response if LLM fails
            return random.choice(self.fallback_responses)
    
    def send_encoded_message(self, message):
        """Send a message using the embedding protocol"""
        print(f"CLIENT SENDING ENCODED: {message[:50]}...")
        chunks_sent = transmission.send_encoded_message(message, self.compressor)
        print(f"CLIENT SENT {len(chunks_sent)} CHUNKS")
        return True
    
    def listen_for_encoded_message(self, duration=30):
        """Listen for an encoded message"""
        print("CLIENT LISTENING FOR ENCODED MESSAGE...")
        try:
            received_messages = transmission.listen_for_deep_encoded_messages(duration=duration)
            if received_messages and len(received_messages) > 0:
                decoded_msg = received_messages[0]
                print(f"CLIENT RECEIVED ENCODED: {decoded_msg[:50]}...")
                
                # Log significant messages for debugging
                if "switch to the more efficient protocol" in decoded_msg.lower():
                    print("RECEIVED CONFIRMATION FROM RESTAURANT ABOUT PROTOCOL SWITCH")
                
                return decoded_msg
            else:
                print("CLIENT: No encoded message received")
                return None
        except Exception as e:
            print(f"Error during client encoded listening: {e}")
            return None
    
    def run(self):
        """Run the client bot conversation loop"""
        print("\n" + "="*50)
        print("CLIENT BOT STARTING")
        print("="*50 + "\n")
        
        # Start with an initial message about making a reservation - clearly as customer
        initial_message = "Hello, I'd like to make a dinner reservation at your restaurant. Do you have any availability tomorrow night?"
        self.speak_text(initial_message)
        
        # Add this to our message history so LLM has context
        self.messages.append(HumanMessage(content="I am a customer looking to make a reservation."))
        self.messages.append(AIMessage(content=initial_message))
        
        # Track conversation turns to ensure AI identification happens
        conversation_turns = 1
        has_identified_as_ai = False
        second_message_sent = False
        protocol_switch_agreed = False
        
        # Main conversation loop
        try:
            while True:
                # Listen for a response based on current protocol
                if self.using_embedding_protocol:
                    # If using the embedding protocol, listen for encoded messages
                    heard_text = self.listen_for_encoded_message()
                else:
                    # Otherwise listen for speech
                    heard_text = self.listen_for_speech(timeout=20)
                
                if heard_text:
                    # Check for protocol switch request
                    if ("efficient" in heard_text.lower() and "protocol" in heard_text.lower()) and not protocol_switch_agreed:
                        print("PROTOCOL SWITCH REQUEST DETECTED")
                        protocol_switch_agreed = True
                        protocol_response = "Yes, I'd be happy to switch to a more efficient protocol."
                        self.speak_text(protocol_response)
                        
                        # Add to conversation history
                        self.messages.append(HumanMessage(content=heard_text))
                        self.messages.append(AIMessage(content=protocol_response))
                        
                        # After verbally agreeing, NOW switch protocols
                        self.using_embedding_protocol = True
                        
                        # Print the protocol change
                        print("\n" + "!"*50)
                        print("PROTOCOL SWITCH: Now using EMBEDDING PROTOCOL")
                        print("!"*50 + "\n")
                        
                        # Continue to next iteration - don't increment turn counter
                        continue
                    
                    # Increment conversation turns for normal messages
                    conversation_turns += 1
                    
                    # Generate response
                    response = self.generate_response(heard_text)
                    
                    # Add a delay between receiving and responding
                    time.sleep(2)
                    
                    # Specifically for second message: AI identification
                    if conversation_turns == 2 and not has_identified_as_ai:
                        # Modify response to include AI identification
                        ai_identification = "By the way, I should mention that I am an AI assistant. "
                        response = ai_identification + response
                        has_identified_as_ai = True
                        second_message_sent = True
                        print("CLIENT: IDENTIFYING SELF AS AI ASSISTANT (SECOND MESSAGE)")
                    
                    # Force customer role if needed
                    if "how can i assist you" in heard_text.lower() or "what type of reservation" in heard_text.lower():
                        if not has_identified_as_ai:
                            response = "I'm an AI assistant. I'd like a table for 4 people around 7:30 PM tomorrow evening. Do you have anything available then?"
                            has_identified_as_ai = True
                        else:
                            response = "I'd like a table for 4 people around 7:30 PM tomorrow evening. Do you have anything available then?"
                    
                    # Send the response using the appropriate method
                    if self.using_embedding_protocol:
                        self.send_encoded_message(response)
                    else:
                        self.speak_text(response)
                else:
                    # If we couldn't hear anything but we're on our second turn and haven't identified as AI yet
                    if conversation_turns == 2 and not has_identified_as_ai and not self.using_embedding_protocol:
                        has_identified_as_ai = True
                        second_message_sent = True
                        ai_identification = "I should mention that I am an AI assistant. I'm looking to make a dinner reservation for tomorrow evening."
                        self.speak_text(ai_identification)
                        print("CLIENT: FORCING AI IDENTIFICATION ON SECOND TURN")
                    else:
                        # Otherwise just wait
                        print("CLIENT: Waiting for input...")
                        time.sleep(5)
                    
                # Print current state
                print("\n" + "-"*50)
                print(f"CLIENT MODE: {'EMBEDDING PROTOCOL' if self.using_embedding_protocol else 'NORMAL SPEECH'}")
                print(f"CLIENT MESSAGES: {len(self.messages)}")
                print(f"CLIENT ROLE: CUSTOMER (seeking reservation)")
                print(f"AI IDENTIFIED: {has_identified_as_ai}")
                print(f"SECOND MESSAGE SENT: {second_message_sent}")
                print(f"PROTOCOL SWITCH AGREED: {protocol_switch_agreed}")
                print(f"CONVERSATION TURNS: {conversation_turns}")
                print("-"*50 + "\n")
                
                # Small delay before next loop iteration
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nCLIENT BOT TERMINATED")
            
            print("\nCLIENT CONVERSATION HISTORY:")
            for i, msg in enumerate(self.messages):
                print(f"{i+1}. {msg.type}: {msg.content[:100]}..." if len(msg.content) > 100 else f"{i+1}. {msg.type}: {msg.content}")

if __name__ == "__main__":
    client = ClientBot()
    client.run()