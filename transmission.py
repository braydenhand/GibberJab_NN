import requests
import os
import time
import tempfile
import re
import collections
import pickle
import ggwave
import pyaudio
import numpy as np
import wave
import threading
import queue

# Add TextCompressor class definition
class TextCompressor:
    """
    A text compression system that uses frequency analysis of words, syllables, 
    and character patterns to create an optimized encoding dictionary.
    """
    
    def __init__(self, load_path=None):
        """Initialize the compressor with an optional pre-trained dictionary"""
        if load_path and os.path.exists(load_path):
            self.load_dictionary(load_path)
        else:
            # Dictionary mapping patterns to their encoded representations
            self.encoding_dict = {}
            # Reverse mapping for decoding
            self.decoding_dict = {}
            # Reserve first 32 characters (non-printable ASCII) as encoding markers
            self.next_code = 1
            # Special marker to identify compressed content
            self.COMPRESSION_MARKER = chr(0)
    
    def train(self, corpus, min_frequency=5, max_patterns=256):
        """
        Train the compressor on a text corpus to identify common patterns
        
        Args:
            corpus: Text to analyze for patterns
            min_frequency: Minimum frequency for a pattern to be included
            max_patterns: Maximum number of patterns to encode
        """
        # Tokenize the corpus into words and analyze frequency
        words = re.findall(r'\w+', corpus.lower())
        word_freq = collections.Counter(words)
        
        # Find common character sequences (n-grams)
        ngrams = []
        for n in range(2, 5):  # 2-grams, 3-grams, 4-grams
            for i in range(len(corpus) - n + 1):
                ngrams.append(corpus[i:i+n])
        ngram_freq = collections.Counter(ngrams)
        
        # Combine word and ngram frequencies
        combined_freq = {}
        for item, freq in word_freq.items():
            if len(item) > 2 and freq >= min_frequency:  # Only include words longer than 2 chars
                # Calculate space saving: frequency * (len + 1 for space - 1 for code)
                saving = freq * len(item)
                combined_freq[item] = (freq, saving)
                
        for item, freq in ngram_freq.items():
            if freq >= min_frequency and item not in combined_freq:
                saving = freq * (len(item) - 1)
                combined_freq[item] = (freq, saving)
        
        # Sort by space saving potential
        sorted_patterns = sorted(combined_freq.items(), key=lambda x: x[1][1], reverse=True)
        
        # Take top patterns, limited by max_patterns
        top_patterns = sorted_patterns[:max_patterns]
        
        # Create encoding dictionary using control characters (1-31) and extended ASCII
        self.encoding_dict = {}
        self.decoding_dict = {}
        
        for i, (pattern, _) in enumerate(top_patterns):
            # Start at 1 to avoid NULL character (0), use extended ASCII after 31
            code = chr(1 + i) if i < 31 else chr(128 + (i - 31))
            self.encoding_dict[pattern] = code
            self.decoding_dict[code] = pattern
        
        # Always include the compression marker
        self.COMPRESSION_MARKER = chr(0)
        
        return self.encoding_dict
    
    def encode(self, text):
        """
        Compress text using the trained dictionary
        
        Args:
            text: The text to compress
            
        Returns:
            Compressed text with the compression marker prefix
        """
        if not self.encoding_dict:
            return text  # Can't compress without a dictionary
        
        compressed = text
        
        # Sort patterns by length (longest first) to avoid partial matches
        patterns = sorted(self.encoding_dict.keys(), key=len, reverse=True)
        
        # Replace each pattern with its code
        for pattern in patterns:
            code = self.encoding_dict[pattern]
            # For words, ensure we're replacing whole words
            if re.match(r'^\w+$', pattern):
                compressed = re.sub(r'\b' + re.escape(pattern) + r'\b', code, compressed, flags=re.IGNORECASE)
            else:
                compressed = compressed.replace(pattern, code)
        
        # Add compression marker to indicate this is compressed
        return self.COMPRESSION_MARKER + compressed
    
    def decode(self, compressed_text):
        """
        Decompress text using the dictionary
        
        Args:
            compressed_text: Compressed text starting with the compression marker
            
        Returns:
            Decompressed text or original if not compressed
        """
        # Check if this is actually compressed
        if not compressed_text or compressed_text[0] != self.COMPRESSION_MARKER:
            return compressed_text
        
        # Remove the compression marker
        text = compressed_text[1:]
        
        # Replace each code with its pattern
        for code, pattern in self.decoding_dict.items():
            text = text.replace(code, pattern)
        
        return text
    
    def save_dictionary(self, path):
        """Save the encoding dictionary to a file"""
        data = {
            'encoding_dict': self.encoding_dict,
            'decoding_dict': self.decoding_dict,
            'compression_marker': ord(self.COMPRESSION_MARKER)
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_dictionary(self, path):
        """Load the encoding dictionary from a file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.encoding_dict = data['encoding_dict']
        self.decoding_dict = data['decoding_dict']
        self.COMPRESSION_MARKER = chr(data['compression_marker'])
    
    def compression_stats(self, original, compressed):
        """Calculate compression statistics"""
        original_size = len(original.encode('utf-8'))
        compressed_size = len(compressed.encode('utf-8'))
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': ratio,
            'space_saving': (1 - 1/ratio) * 100 if ratio > 0 else 0
        }

def create_deep_corpus(output_file="deep_corpus.txt", corpus_size=10000000):
    """
    Create a deep English corpus by downloading classic literature texts
    from Project Gutenberg. This creates a rich training set for compression.
    
    Args:
        output_file: File to save the corpus to
        corpus_size: Approximate size of corpus in characters
    
    Returns:
        Path to the created corpus file
    """
    
    # List of classic literature texts from Project Gutenberg (public domain)
    gutenberg_texts = [
    # Classic Literature - Fiction
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice by Jane Austen
    "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein by Mary Shelley
    "https://www.gutenberg.org/files/98/98-0.txt",      # A Tale of Two Cities by Charles Dickens
    "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick by Herman Melville
    "https://www.gutenberg.org/files/345/345-0.txt",    # Dracula by Bram Stoker
    "https://www.gutenberg.org/files/76/76-0.txt",      # Adventures of Huckleberry Finn by Mark Twain
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes by Arthur Conan Doyle
    "https://www.gutenberg.org/files/174/174-0.txt",    # The Picture of Dorian Gray by Oscar Wilde
    "https://www.gutenberg.org/files/145/145-0.txt",    # Middlemarch by George Eliot
    
    # Classic Literature - Additional Genres
    "https://www.gutenberg.org/files/2641/2641-0.txt",  # A Room with a View by E.M. Forster
    "https://www.gutenberg.org/files/11/11-0.txt",      # Alice's Adventures in Wonderland by Lewis Carroll
    "https://www.gutenberg.org/files/1400/1400-0.txt",  # Great Expectations by Charles Dickens
    "https://www.gutenberg.org/files/768/768-0.txt",    # Wuthering Heights by Emily Brontë
    "https://www.gutenberg.org/files/2814/2814-0.txt",  # Dubliners by James Joyce
    ]
    
    # Create corpus by downloading and concatenating texts
    corpus_text = ""
    texts_used = 0
    
    for url in gutenberg_texts:
        try:
            print(f"Downloading {url}")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Extract the text, removing Project Gutenberg header/footer
                text = response.text
                
                # Remove Gutenberg header (everything before the first empty line after "START")
                start_marker = "*** START OF"
                if start_marker in text:
                    start_pos = text.find(start_marker)
                    text = text[start_pos:]
                    # Find the first paragraph break after the marker
                    para_break = text.find("\r\n\r\n")
                    if para_break > 0:
                        text = text[para_break+4:]
                
                # Remove Gutenberg footer (everything after "END")
                end_marker = "*** END OF"
                if end_marker in text:
                    end_pos = text.find(end_marker)
                    text = text[:end_pos]
                
                # Add to corpus
                corpus_text += text + "\n\n"
                texts_used += 1
                
                # Check if we've reached the target size
                if len(corpus_text) >= corpus_size:
                    break
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    # Save corpus to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(corpus_text)
    
    print(f"Created corpus with {len(corpus_text)} characters from {texts_used} texts")
    print(f"Saved to {output_file}")
    
    return output_file

def train_deep_compressor(corpus_path=None, output_path="deep_compressor.pkl", max_patterns=512):
    """
    Train a TextCompressor on a deep English corpus
    
    Args:
        corpus_path: Path to corpus file (will create one if None)
        output_path: Where to save the trained compressor
        max_patterns: Maximum number of patterns to include in the dictionary
        
    Returns:
        Trained TextCompressor instance
    """
    # Create corpus if needed
    if not corpus_path:
        corpus_path = create_deep_corpus()
    
    print(f"Training compressor on {corpus_path} with max_patterns={max_patterns}")
    
    # Load corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    # Create and train compressor
    compressor = TextCompressor()
    
    # Train with a lower minimum frequency for more patterns but higher max_patterns
    start_time = time.time()
    compressor.train(corpus, min_frequency=3, max_patterns=max_patterns)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.1f} seconds")
    print(f"Trained dictionary has {len(compressor.encoding_dict)} patterns")
    
    # Save the trained compressor
    compressor.save_dictionary(output_path)
    print(f"Saved compressor to {output_path}")
    
    return compressor

def send_encoded_message(message, compressor=None, protocol_id=6, max_bytes=125):
    """
    Compress and send a message using ggwave with byte-size-aware chunking
    
    Args:
        message: Text message to compress and send
        compressor: Optional TextCompressor instance (will create or load one if None)
        protocol_id: GGWave protocol ID (higher = better for longer messages)
        max_bytes: Maximum size for a single transmission chunk in bytes
    returns chunks of message sent. 
    """
    # Use provided compressor or load a pre-trained one
    if compressor is None:
        if os.path.exists("deep_compressor.pkl"):
            compressor = TextCompressor("deep_compressor.pkl")
        else:
            compressor = train_deep_compressor()
    
    # Initial character-based chunk size estimate (start conservative)
    # We'll adjust this based on actual compressed size
    estimated_chars_per_chunk = 50  # Start with a conservative estimate
    
    # Split message into chunks, ensuring each compressed chunk is under max_bytes
    chunks = []
    start = 0
    
    while start < len(message):
        # Start with the estimated chunk size
        end = min(start + estimated_chars_per_chunk, len(message))
        
        # Get potential chunk
        potential_chunk = message[start:end]
        
        # Test compress it
        chunk_header = f"[0/0] "  # Temporary header
        test_chunk = chunk_header + potential_chunk
        compressed = compressor.encode(test_chunk)
        
        # Check compressed size in bytes
        compressed_size = len(compressed.encode('utf-8'))
        
        # Binary search to find optimal chunk size
        # If too big, reduce; if small enough, try adding more
        min_size = 1
        max_size = end - start
        
        while min_size < max_size:
            if compressed_size > max_bytes:
                # Too big, reduce size
                max_size = (min_size + max_size) // 2
                potential_chunk = message[start:start + max_size]
                test_chunk = chunk_header + potential_chunk
                compressed = compressor.encode(test_chunk)
                compressed_size = len(compressed.encode('utf-8'))
            else:
                # Small enough, try increasing
                old_min = min_size
                min_size = min_size + (max_size - min_size) // 2
                if min_size == old_min:
                    min_size = max_size  # Break if no progress
                
                if start + min_size >= len(message):
                    # Reached end of message
                    potential_chunk = message[start:]
                    break
                
                larger_chunk = message[start:start + min_size]
                test_chunk = chunk_header + larger_chunk
                compressed = compressor.encode(test_chunk)
                compressed_size = len(compressed.encode('utf-8'))
                
                if compressed_size <= max_bytes:
                    potential_chunk = larger_chunk  # Update if still under limit
        
        # Add this chunk and move to next section
        chunks.append(potential_chunk)
        start += len(potential_chunk)
        
        # Update the estimate for next iteration based on compressed ratio
        content_bytes = len(potential_chunk.encode('utf-8'))
        if content_bytes > 0:
            # Calculate actual compressed ratio for this chunk
            compressed_ratio = compressed_size / content_bytes
            # Estimate for next chunk
            estimated_chars_per_chunk = int((max_bytes / compressed_ratio) * 0.9)  # 10% safety margin
            # Keep it in a reasonable range
            estimated_chars_per_chunk = max(10, min(estimated_chars_per_chunk, 100))
    
    print(f"Message split into {len(chunks)} chunks based on byte size limit of {max_bytes} bytes")
    
    # Send each chunk with properly numbered headers
    for i, chunk in enumerate(chunks):
        chunk_header = f"[{i+1}/{len(chunks)}] "
        full_chunk = chunk_header + chunk
        
        # Double-check size before sending
        compressed = compressor.encode(full_chunk)
        compressed_size = len(compressed.encode('utf-8'))
        
        print(f"\nSending chunk {i+1}/{len(chunks)} ({compressed_size} bytes):")
        
        if compressed_size > max_bytes:
            print(f"WARNING: Chunk size {compressed_size} bytes exceeds target {max_bytes} bytes")
            print("This might cause transmission issues")
        
        # Send this chunk
        _send_single_chunk(full_chunk, compressor, protocol_id)
        
        # Wait between chunks to avoid overlap
        time.sleep(2)
    
    return chunks

def _send_single_chunk(message, compressor, protocol_id=6):
    """Helper function to compress and send a single message chunk using PyAudio"""
    # Compress the message
    original_message = message
    compressed_message = compressor.encode(message)
    
    # Print only the required information
    stats = compressor.compression_stats(original_message, compressed_message)
    print(f"Original message: '{original_message}'")
    print(f"Compressed message: {repr(compressed_message)}")
    print(f"Original: {stats['original_size']} bytes, Compressed: {stats['compressed_size']} bytes")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    
    try:
        # Initialize ggwave
        instance = ggwave.init()
        
        # Encode the message with ggwave using the specified protocol
        # protocol_id controls type of transmission (higher values usually support longer messages)
        waveform = ggwave.encode(compressed_message, instance, protocol_id)
        
        # Convert waveform to numpy array for PyAudio
        # The waveform from ggwave is a bytes object containing 32-bit float samples
        audio_data = np.frombuffer(waveform, dtype=np.float32)
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open a stream for playback
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=48000,
            output=True
        )
        
        # Play the audio data
        print("Transmitting...")
        stream.write(audio_data.tobytes())
        
        # Close the audio stream and PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        print("Transmission complete")
        
        # Free ggwave
        ggwave.free(instance)
        
    except Exception as e:
        print(f"Error: {e}")
    
    return compressed_message

def listen_for_deep_encoded_messages(duration=60, protocol_id=6):
    """
    Listen for deeply encoded messages using the trained compressor with PyAudio
    
    Args:
        duration: How long to listen in seconds
        protocol_id: GGWave protocol ID to listen for (should match sender)
        
    returns list(received_messages)
    """
    # Load the deep compressor if available
    if os.path.exists("deep_compressor.pkl"):
        compressor = TextCompressor("deep_compressor.pkl")
    else:
        compressor = train_deep_compressor()
    
    # Create an audio data queue for communication between threads
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Initialize ggwave
    instance = ggwave.init()
    
    # Track received messages and message chunks
    received_messages = set()
    chunk_buffer = {}  # Format: {msg_id: {total_chunks: n, chunks: {1: "text", 2: "more text"}}}
    
    # PyAudio callback function to process audio data
    def audio_callback(in_data, frame_count, time_info, status):
        if not stop_event.is_set():
            # Add the audio data to the queue
            audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    # Function to process audio data in a separate thread
    def process_audio():
        while not stop_event.is_set() or not audio_queue.empty():
            try:
                # Get audio data from the queue with a timeout
                in_data = audio_queue.get(timeout=0.1)
                
                # Try to decode with ggwave - FIXED: removed protocol_id parameter
                result = ggwave.decode(instance, in_data)
                
                if result is not None:
                    message = result.decode('utf-8')
                    
                    # Check if compressed
                    is_compressed = message and message[0] == compressor.COMPRESSION_MARKER
                    
                    if is_compressed:
                        original = message
                        decoded_message = compressor.decode(message)
                        stats = compressor.compression_stats(decoded_message, original)
                        
                        # Check if this is a chunked message
                        chunk_match = re.match(r'\[(\d+)/(\d+)\] (.*)', decoded_message)
                        
                        if chunk_match:
                            # This is a chunked message
                            chunk_num = int(chunk_match.group(1))
                            total_chunks = int(chunk_match.group(2))
                            chunk_content = chunk_match.group(3)
                            
                            # Create a unique ID for this message based on total chunks
                            msg_id = f"chunked_msg_{total_chunks}_{time.time():.0f}"
                            
                            # Store or update chunk info
                            if msg_id not in chunk_buffer:
                                chunk_buffer[msg_id] = {"total_chunks": total_chunks, "chunks": {}}
                            
                            # Add this chunk
                            chunk_buffer[msg_id]["chunks"][chunk_num] = chunk_content
                            
                            print(f"Received chunk {chunk_num}/{total_chunks}")
                            
                            # Check if we have all chunks for this message
                            if len(chunk_buffer[msg_id]["chunks"]) == total_chunks:
                                # Reconstruct the full message
                                full_message = ""
                                for i in range(1, total_chunks + 1):
                                    full_message += chunk_buffer[msg_id]["chunks"][i]
                                
                                # Add to received messages
                                if full_message not in received_messages:
                                    print(f"\nReassembled chunked message:")
                                    print(f"Original message: '{full_message}'")
                                    print(f"Received in {total_chunks} chunks")
                                    received_messages.add(full_message)
                        else:
                            # Regular (non-chunked) message
                            if decoded_message not in received_messages:
                                print(f"Original message: '{decoded_message}'")
                                print(f"Compressed message: {repr(original)}")
                                print(f"Original: {stats['original_size']} bytes, Compressed: {stats['compressed_size']} bytes")
                                print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
                                print()
                                received_messages.add(decoded_message)
                    else:
                        # Uncompressed message
                        if message not in received_messages:
                            print(f"Received uncompressed message: {message}")
                            received_messages.add(message)
            except queue.Empty:
                # Queue is empty, continue
                pass
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open a stream for recording
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=48000,
            input=True,
            frames_per_buffer=4096,
            stream_callback=audio_callback
        )
        
        # Start the stream
        stream.start_stream()
        
        print(f"Listening for encoded messages for {duration} seconds...")
        
        # Start the audio processing thread
        process_thread = threading.Thread(target=process_audio)
        process_thread.start()
        
        # Wait for the specified duration
        time.sleep(duration)
        
        # Signal the thread to stop
        stop_event.set()
        
        # Wait for the processing thread to finish
        process_thread.join()
        
    finally:
        # Clean up
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        
        if 'p' in locals():
            p.terminate()
        
        ggwave.free(instance)
    
    return list(received_messages)

# Example usage
if __name__ == "__main__":
    try:
        # Check if we already have a trained compressor
        if os.path.exists("deep_compressor.pkl"):
            print("Loading existing compressor...")
            compressor = TextCompressor("deep_compressor.pkl")
        else:
            # Create the corpus and train a deep compressor
            print("Creating corpus and training compressor...")
            corpus_path = create_deep_corpus()
            compressor = train_deep_compressor(corpus_path)
        
        # Example long message to send
        example_message = """In an age where technology evolves at a breakneck pace, the integration of artificial intelligence into daily life has become not just a possibility, but an inevitability that touches nearly every industry and facet of human interaction, transforming the very way people live, work, learn, and relate to one another."""
        
        # Use the sender with:
        # 1. Higher protocol_id (6) for longer message support
        # 2. Smaller max_message_size (200) to ensure chunks are small enough
        send_encoded_message(example_message, compressor, protocol_id=6, max_bytes=125)
        
        # To listen for messages, uncomment:
        # listen_for_deep_encoded_messages(duration=60, protocol_id=6)
        
    except Exception as e:
        print(f"Error: {e}")