import ggwave
import subprocess
import tempfile
import os
import sys

def send_message(message):
    """
    Send a message using ggwave audio transmission via SoX
    """
    print(f"Sending message: '{message}'")
    
    # Initialize ggwave
    instance = ggwave.init()
    
    try:
        # Encode the message
        waveform = ggwave.encode(message)
        
        # Create a temporary RAW file
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
            temp_filename = temp_file.name
            
            # Write bytes directly to file
            temp_file.write(waveform)
            temp_file.flush()
        
        # Play the audio file using 'play' command from SoX with correct format parameters
        print("Playing transmission...")
        subprocess.run([
            'play', 
            '-t', 'f32', 
            '-r', '48000',  # sample rate
            '-c', '1',      # channels
            '-b', '32',     # bits
            temp_filename
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Transmission complete")
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
    except Exception as e:
        print(f"Error during transmission: {e}")
    
    finally:
        # Clean up resources
        ggwave.free(instance)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <message>")
        sys.exit(1)
    
    message = " ".join(sys.argv[1:])
    send_message(message)
