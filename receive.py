import ggwave
import subprocess
import tempfile
import os
import time

def listen_for_messages(duration=60):
    """
    Listen for GGWave messages for the specified duration (in seconds)
    """
    print(f"Listening for {duration} seconds...")
    
    # Initialize ggwave
    instance = ggwave.init()
    
    # To keep track of already received messages
    received_messages = set()
    
    try:
        # Create a temporary file for SoX to write to
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # Start recording in the background
        sox_process = subprocess.Popen([
            'rec',
            '-q',                # Quiet mode
            '-t', 'f32',         # Same format as sender
            '-r', '48000',       # Same sample rate as sender
            '-c', '1',           # Mono
            '-b', '32',          # 32-bit
            temp_filename
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait and check for messages
        start_time = time.time()
        end_time = start_time + duration
        last_check_size = 0
        
        while time.time() < end_time:
            # Sleep to allow recording to accumulate
            time.sleep(1)
            
            try:
                # Get the current size of the recording file
                current_size = os.path.getsize(temp_filename)
                
                # Only process if there's new data
                if current_size > last_check_size:
                    # Read the current file content
                    with open(temp_filename, 'rb') as f:
                        # Only read the new data
                        f.seek(last_check_size)
                        new_data = f.read()
                    
                    # Update the last checked size
                    last_check_size = current_size
                    
                    # Try to decode
                    result = ggwave.decode(instance, new_data)
                    
                    # If we got a message, print it (but only if new)
                    if result is not None:
                        message = result.decode('utf-8')
                        
                        # Only print if we haven't seen this message before
                        if message not in received_messages:
                            print(f"Received: \"{message}\"")
                            received_messages.add(message)
            except Exception as e:
                # Quietly handle errors
                pass
                
    except KeyboardInterrupt:
        print("\nStopped listening.")
    finally:
        # Clean up
        if 'sox_process' in locals():
            sox_process.terminate()
        
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.unlink(temp_filename)
        
        ggwave.free(instance)
        
        if not received_messages:
            print("No messages received.")
        else:
            print(f"Listening complete. Received {len(received_messages)} unique messages.")

# Start listening for 60 seconds
listen_for_messages(60)