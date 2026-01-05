"""
Voice-to-Voice RAG Pipeline
============================
Part 1: Deepgram (Speech-to-Text) - Microphone input
Part 1.5: Milvus RAG (Retrieval) - Search relevant documents
Part 2: Gemini 2.5 Flash (LLM) - Generate response with context
Part 3: Deepgram (Text-to-Speech) - Streaming Audio output

Requirements:
- deepgram-sdk==3.7.0 (for LiveTranscriptionEvents support)
- websocket-client
- google-generativeai
- pyaudio
- pymilvus
- sentence-transformers
"""

import asyncio
import os
import time
import json
import wave
import threading
from queue import Queue
from typing import Optional
import pyaudio
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

try:
    import websocket
except ImportError:
    print("Please install websocket-client: pip install websocket-client")
    exit(1)

# Load environment variables
load_dotenv()

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Audio settings for microphone
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16

# TTS Configuration
TTS_MODEL = "aura-2-thalia-en"  # Voice model
TTS_ENCODING = "linear16"        # PCM 16-bit
TTS_SAMPLE_RATE = 24000          # 24kHz
OUTPUT_FILE = "response.wav"
HISTORY_FILE = "conversation_history.json"

# Streaming behavior
FLUSH_ON_SENTENCE = True  # Flush TTS buffer after each sentence
SENTENCE_ENDINGS = (".", "!", "?", "\n")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Milvus Configuration
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_DB = "DEV_DB"
MILVUS_COLLECTION = "company_abc123"
TOP_K = 5  # Number of chunks to retrieve


# =============================================================================
# Audio Player (real-time playback)
# =============================================================================

class AudioPlayer:
    """Handles real-time audio playback using PyAudio."""
    
    def __init__(self, sample_rate: int = TTS_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.audio_queue: Queue = Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.p: Optional[pyaudio.PyAudio] = None
        self.stream = None
        
    def start(self):
        """Start the audio playback thread."""
        self.running = True
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=1024
        )
        self.thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.thread.start()
        
    def _playback_loop(self):
        """Background thread that plays audio chunks."""
        while self.running or not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk and self.stream:
                    self.stream.write(chunk)
            except:
                continue
                
    def play(self, audio_data: bytes):
        """Add audio data to playback queue."""
        if self.running:
            self.audio_queue.put(audio_data)
    
    def wait_until_done(self, timeout: float = 60):
        """Wait for all queued audio to finish playing."""
        start_time = time.time()
        while not self.audio_queue.empty():
            if time.time() - start_time > timeout:
                print("  ⚠ Audio playback timeout")
                break
            time.sleep(0.1)
        
        # Give a little extra time for the last chunk to play
        time.sleep(0.5)
            
    def stop(self):
        """Stop playback and clean up."""
        # Wait for queue to drain first
        self.wait_until_done()
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()


# =============================================================================
# WAV File Writer
# =============================================================================

class WavWriter:
    """Writes PCM audio data to a WAV file."""
    
    def __init__(self, filename: str, sample_rate: int = TTS_SAMPLE_RATE, 
                 channels: int = 1, sample_width: int = 2):
        self.filename = filename
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.audio_data = bytearray()
        
    def write(self, data: bytes):
        """Append audio data."""
        self.audio_data.extend(data)
        
    def save(self):
        """Save accumulated audio to WAV file."""
        if not self.audio_data:
            return
            
        with wave.open(self.filename, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(bytes(self.audio_data))
            
        print(f"✓ Audio saved to {self.filename}")


# =============================================================================
# Deepgram TTS WebSocket Client
# =============================================================================

class DeepgramTTS:
    """Streaming TTS client using Deepgram's WebSocket API."""
    
    def __init__(self, api_key: str, model: str = TTS_MODEL, 
                 encoding: str = TTS_ENCODING, sample_rate: int = TTS_SAMPLE_RATE):
        self.api_key = api_key
        self.model = model
        self.encoding = encoding
        self.sample_rate = sample_rate
        
        self.ws: Optional[websocket.WebSocket] = None
        self.wav_writer = WavWriter(OUTPUT_FILE, sample_rate)
        self.audio_player = AudioPlayer(sample_rate)
        
        self.connected = False
        self.receive_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Event flags
        self.flushed_event = threading.Event()
        self.closed_event = threading.Event()
        
        # Stats
        self.text_sent = 0
        self.audio_received = 0
        
    @property
    def url(self) -> str:
        """Build the WebSocket URL with parameters."""
        return (
            f"wss://api.deepgram.com/v1/speak"
            f"?model={self.model}"
            f"&encoding={self.encoding}"
            f"&sample_rate={self.sample_rate}"
        )
        
    def connect(self) -> bool:
        """Establish WebSocket connection to Deepgram."""
        try:
            print(f"Connecting to Deepgram TTS...")
            
            self.ws = websocket.create_connection(
                self.url,
                header={"Authorization": f"Token {self.api_key}"}
            )
            
            self.connected = True
            self.running = True
            
            # Start audio player
            self.audio_player.start()
            
            # Start receive thread
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            
            print("✓ Connected to Deepgram TTS\n")
            return True
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
            
    def _receive_loop(self):
        """Background thread to receive audio and messages from WebSocket."""
        while self.running and self.ws:
            try:
                data = self.ws.recv()
                
                if isinstance(data, bytes):
                    # Binary audio data
                    self.audio_received += len(data)
                    self.wav_writer.write(data)
                    self.audio_player.play(data)
                    
                elif isinstance(data, str):
                    # Skip empty strings
                    if not data.strip():
                        continue
                        
                    # JSON message
                    try:
                        message = json.loads(data)
                        msg_type = message.get("type", "Unknown")
                        
                        if msg_type == "Flushed":
                            self.flushed_event.set()
                        elif msg_type == "Warning":
                            print(f"  ⚠ TTS Warning: {message.get('warn_msg', 'Unknown')}")
                        elif msg_type == "Error":
                            print(f"  ✗ TTS Error: {message.get('err_msg', 'Unknown')}")
                    except json.JSONDecodeError:
                        pass
                        
            except websocket.WebSocketConnectionClosedException:
                break
            except Exception as e:
                if self.running and self.connected:
                    print(f"  TTS receive error: {e}")
                break
                
        self.running = False
        self.closed_event.set()
        
    def send_text(self, text: str) -> bool:
        """Send text to be converted to speech."""
        if not self.ws or not self.connected:
            return False
        
        try:
            message = json.dumps({"type": "Speak", "text": text})
            self.ws.send(message)
            self.text_sent += len(text)
            return True
        except Exception as e:
            print(f"\n  ✗ TTS send error: {e}")
            self.connected = False
            return False
        
    def flush(self) -> bool:
        """Send Flush command."""
        if not self.ws or not self.connected:
            return False
        
        try:
            self.flushed_event.clear()
            self.ws.send(json.dumps({"type": "Flush"}))
            return True
        except Exception:
            self.connected = False
            return False
        
    def wait_for_flush(self, timeout: float = 30):
        """Wait for Flushed response."""
        return self.flushed_event.wait(timeout)
        
    def close(self):
        """Close the WebSocket connection gracefully."""
        was_connected = self.connected
        self.connected = False
        self.running = False
        
        if self.ws and was_connected:
            try:
                self.ws.send(json.dumps({"type": "Close"}))
                self.closed_event.wait(timeout=2)
            except:
                pass
        
        self.audio_player.stop()
        self.wav_writer.save()
        
        if self.ws:
            try:
                self.ws.close()
            except:
                pass


class VoicePipeline:
    def __init__(self):
        self.deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        self.tts = DeepgramTTS(api_key=DEEPGRAM_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        self.transcribed_text = ""
        self.final_transcript = ""
        self.is_listening = True
        self.silence_timeout = 1.0  # seconds of silence before processing
        self.last_transcript_time = None
        
        # Initialize embedding model (BAAI/bge-large-en-v1.5 produces 1024 dimensions)
        print("\n📦 Loading embedding model (BAAI/bge-large-en-v1.5 - 1024 dim)...")
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Connect to Milvus
        print("🗄️ Connecting to Milvus...")
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, db_name=MILVUS_DB)
        self.collection = Collection(MILVUS_COLLECTION)
        self.collection.load()
        self.collection = Collection(MILVUS_COLLECTION)
        self.collection.load()
        print(f"✅ Connected to collection: {MILVUS_COLLECTION}\n")
        
        # Initialize history
        self.history_file = HISTORY_FILE
        self.history = self.load_history()
        
    def load_history(self):
        """Load conversation history from JSON file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠ Error loading history: {e}")
        return []

    def save_history(self):
        """Save conversation history to JSON file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving history: {e}")
        
    def speech_to_text(self):
        """Part 1: Use Deepgram to convert speech to text from microphone"""
        
        # Create live transcription connection
        connection = self.deepgram_client.listen.live.v("1")
        
        # Event handler for transcription results
        def on_message(self_conn, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if transcript.strip():
                # Check if this is a final result
                if result.is_final:
                    self.final_transcript += " " + transcript
                    self.final_transcript = self.final_transcript.strip()
                self.transcribed_text = transcript
                self.last_transcript_time = time.time()
                print(f"🎤 User: {transcript}")
        
        def on_error(self_conn, error, **kwargs):
            print(f"❌ Deepgram Error: {error}")
        
        def on_close(self_conn, close, **kwargs):
            print("🔌 Deepgram connection closed")
        
        def on_utterance_end(self_conn, utterance_end, **kwargs):
            print("\n📝 Utterance ended - Processing...")
            self.is_listening = False
        
        # Register event handlers
        connection.on(LiveTranscriptionEvents.Transcript, on_message)
        connection.on(LiveTranscriptionEvents.Error, on_error)
        connection.on(LiveTranscriptionEvents.Close, on_close)
        connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        
        # Configure transcription options
        options = LiveOptions(
            model="nova-3",
            language="en-IN",
            smart_format=True,
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            interim_results=True,
            endpointing=500,  # milliseconds of silence for endpoint detection
            utterance_end_ms=1500  # end of utterance detection
        )
        
        # Start the connection
        if connection.start(options) is False:
            print("❌ Failed to start Deepgram connection")
            return None
        
        print("🎙️ Listening... Speak now! (Press Ctrl+C to stop)")
        
        # Initialize PyAudio for microphone input
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        try:
            while self.is_listening:
                # Read audio from microphone
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                connection.send(data)
                
                # Check for silence timeout (user stopped speaking)
                if self.transcribed_text and self.last_transcript_time:
                    current_time = time.time()
                    if current_time - self.last_transcript_time > self.silence_timeout:
                        print("\n⏸️ Processing your query...")
                        break
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            connection.finish()
        
        # Return final accumulated transcript
        return self.final_transcript if self.final_transcript else self.transcribed_text
    
    def retrieve_context(self, query: str) -> list:
        """Part 1.5: Retrieve relevant chunks from Milvus using vector search"""
        
        print(f"\n🔍 Searching knowledge base for query: '{query}'")
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query).tolist()
        print(f"  📊 Generated embedding (first 5 dim): {query_embedding[:5]}")
        
        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Perform vector search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="vector",  # Adjust field name if different
            param=search_params,
            limit=TOP_K,
            output_fields=["text"]  # Adjust field name for text content
        )
        
        # Extract retrieved chunks
        retrieved_chunks = []
        for hits in results:
            for i, hit in enumerate(hits):
                chunk_text = hit.entity.get("text", "")
                if chunk_text:
                    retrieved_chunks.append(chunk_text)
                    print(f"  📄 Chunk {i+1}: {chunk_text[:100]}...")
        
        print(f"  ✅ Retrieved {len(retrieved_chunks)} relevant chunks")
        return retrieved_chunks
    
    async def generate_and_stream_response(self, user_query: str, context_chunks: list = None):
        """Part 2 & 3: Generate response with Gemini and stream to Deepgram TTS"""
        
        if not user_query:
            print("I didn't catch that. Could you please repeat?")
            return
        
        print(f"\n🤔 Thinking...")
        
        # Build context from retrieved chunks
        context = ""
        if context_chunks:
            context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # System prompt with RAG context
        system_prompt = """You are a helpful voice assistant with access to a knowledge base.
        Use the provided context to answer the user's question accurately.
        If the context doesn't contain relevant information, say so and provide what help you can.
        Keep your responses concise and conversational since they will be spoken aloud.
        Avoid using markdown, bullet points, or special formatting."""
        
        # Build the full prompt
        if context:
            full_prompt = f"{system_prompt}\n\n--- KNOWLEDGE BASE CONTEXT ---\n{context}\n\n--- USER QUESTION ---\n{user_query}"
        else:
            full_prompt = f"{system_prompt}\n\nUser Query: {user_query}"
        
        # Connect to TTS
        if not self.tts.connect():
            print("❌ Failed to connect to TTS service")
            return

        try:
            print(f"🤖 Assistant (streaming):")
            
            # Generate response using Gemini (streaming)
            
            # Format history for prompt
            history_text = ""
            if self.history:
                history_text = "\n\n--- CONVERSATION HISTORY ---\n"
                for turn in self.history[-5:]: # Keep last 5 turns context
                    history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            
            # Build the full prompt with history
            final_prompt = f"{full_prompt}{history_text}\n\nAssistant:"
            
            # Note: generate_content_async with stream=True returns an async iterator
            response = await self.gemini_model.generate_content_async(final_prompt, stream=True)
            
            text_buffer = ""
            full_response_text = ""
            
            async for chunk in response:
                if chunk.text:
                    text_segment = chunk.text
                    full_response_text += text_segment
                    print(text_segment, end="", flush=True)
                    
                    # Send to TTS
                    self.tts.send_text(text_segment)
                    
                    # Check for sentence boundaries to flush
                    if FLUSH_ON_SENTENCE:
                        text_buffer += text_segment
                        if any(text_buffer.rstrip().endswith(ending) for ending in SENTENCE_ENDINGS):
                            self.tts.flush()
                            # We don't wait for flush here to keep streaming fast, 
                            # but the TTS client handles the queue.
                            text_buffer = ""
            
            print("\n")
            
            # Final flush
            if text_buffer.strip():
                self.tts.flush()
            
            # Wait for audio to finish playing
            self.tts.wait_for_flush(timeout=10)
            # Give a little extra time for the player to drain
            time.sleep(1.0)
            
            # Save to history
            self.history.append({
                "user": user_query,
                "assistant": full_response_text,
                "timestamp": time.time()
            })
            self.save_history()
            
        except Exception as e:
            print(f"\n❌ Error during generation/TTS: {e}")
        finally:
            self.tts.close()
    
    async def run_pipeline(self):
        """Run the complete voice-to-voice RAG pipeline"""
        
        print("\n" + "="*50)
        print("🎯 VOICE-TO-VOICE RAG PIPELINE")
        print("="*50)
        print("Part 1: Deepgram (Speech-to-Text)")
        print("Part 1.5: Milvus RAG (Retrieval)")
        print("Part 2: Gemini 2.0 Flash (LLM)")
        print("Part 3: Deepgram (Text-to-Speech)")
        print("="*50 + "\n")
        
        print("="*50 + "\n")
        
        while True:
            print("\nReady for new query... (Say 'exit' or 'stop' to quit)")
            
            # Reset state for new turn
            self.transcribed_text = ""
            self.final_transcript = ""
            self.is_listening = True
            
            # Part 1: Speech to Text (synchronous)
            user_query = self.speech_to_text()
            
            if not user_query:
                print("No speech detected. Continuing...")
                continue
                
            # Check for exit commands
            if user_query.lower().strip().rstrip('.').rstrip('!') in ["exit", "stop", "quit", "bye"]:
                print("👋 Exiting pipeline. Goodbye!")
                break
            
            # Part 1.5: Retrieve relevant context from Milvus
            context_chunks = self.retrieve_context(user_query)
            
            # Part 2 & 3: Generate response and stream TTS
            await self.generate_and_stream_response(user_query, context_chunks)
            
            print("\n✨ Turn complete!")


async def main():
    pipeline = VoicePipeline()
    await pipeline.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
