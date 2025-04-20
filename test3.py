import ggwave
import pyaudio
import time

def listen_for_messages(duration=60):
    """
    Listen for GGWave messages for the specified duration (in seconds)
    using PyAudio instead of SoX.
    """
    print(f"Listening for {duration} seconds...")

    # GGWave expects:
    SAMPLE_RATE = 48000
    SAMPLE_FORMAT = pyaudio.paFloat32  # 32-bit float
    CHANNELS = 1
    CHUNK = 1024  # Buffer size per read (tweak for latency vs CPU)

    # Initialize GGWave and PyAudio
    ggwave_instance = ggwave.init()
    received_messages = set()
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=SAMPLE_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    try:
        start_time = time.time()
        print("Listening... Press Ctrl+C to stop early.")
        while time.time() - start_time < duration:
            # Read a chunk of raw audio data
            data = stream.read(CHUNK, exception_on_overflow=False)

            # Decode using GGWave
            result = ggwave.decode(ggwave_instance, data)
            if result:
                try:
                    message = result.decode("utf-8")
                    if message not in received_messages:
                        print(f"Received: \"{message}\"")
                        received_messages.add(message)
                except UnicodeDecodeError:
                    # Occasionally garbled noise might sneak in
                    continue

    except KeyboardInterrupt:
        print("\nStopped listening.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        ggwave.free(ggwave_instance)

        if not received_messages:
            print("No messages received.")
        else:
            print(f"Listening complete. Received {len(received_messages)} unique messages.")
listen_for_messages()