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
- PyPDF2>=3.0.0
- pdfplumber>=0.10.0
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
from pdf_processor import PDFProcessor, select_pdf_file

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
MILVUS_DB = "TEST_VOICE"  # Updated to new database
MILVUS_COLLECTION = "pdf_knowledge_base"  # Updated to new collection
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
        # Configure Gemini for fast, concise voice responses
        generation_config = genai.GenerationConfig(
            temperature=0.7,
        )
        self.gemini_model = genai.GenerativeModel(
            'gemini-2.5-flash-lite',
            generation_config=generation_config
        )
        self.transcribed_text = ""
        self.final_transcript = ""
        self.is_listening = True
        self.silence_timeout = 2.0  # seconds of silence before processing (balanced for pauses and responsiveness)
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
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(chunk_size=800, chunk_overlap=150)
        
        # TTS connection state
        self.tts_connected = False
        
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
        
    def upload_and_store_pdf(self, pdf_path: str) -> bool:
        """
        Upload PDF, process it, and store chunks in Milvus.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print("\n" + "="*50)
            print("📤 PDF UPLOAD AND PROCESSING")
            print("="*50)
            
            # Process PDF to chunks
            chunks, metadata = self.pdf_processor.process_pdf_to_chunks(pdf_path)
            
            if not chunks:
                print("❌ No content extracted from PDF")
                return False
            
            # Prepare data for Milvus
            milvus_data = self.pdf_processor.prepare_for_milvus(
                chunks, metadata, self.embedding_model
            )
            
            # Insert into Milvus
            print(f"\n💾 Storing {len(milvus_data)} chunks in Milvus...")
            
            # Prepare data in Milvus format (id is auto-generated, don't include it)
            texts = [item['text'] for item in milvus_data]
            vectors = [item['vector'] for item in milvus_data]
            source_types = [item['source_type'] for item in milvus_data]
            filenames = [item['filename'] for item in milvus_data]
            page_numbers = [item['page_number'] for item in milvus_data]
            timestamps = [item['upload_timestamp'] for item in milvus_data]
            
            # Insert data (order must match schema: text, vector, source_type, filename, page_number, upload_timestamp)
            insert_data = [
                texts,
                vectors,
                source_types,
                filenames,
                page_numbers,
                timestamps
            ]
            
            self.collection.insert(insert_data)
            self.collection.flush()
            
            print(f"  ✅ Successfully stored {len(milvus_data)} chunks")
            print(f"  📊 PDF: {metadata['filename']}")
            print(f"  📄 Pages: {metadata['total_pages']}")
            print(f"  🔢 Chunks: {metadata['total_chunks']}")
            print("\n" + "="*50)
            print("✅ PDF UPLOAD COMPLETE")
            print("="*50 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error uploading PDF: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
            endpointing=2500,  # milliseconds of silence for endpoint detection (2.5 seconds - balanced)
            utterance_end_ms=2000  # end of utterance detection (2 seconds - balanced)
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
        
        print(f"\n🔍 Searching knowledge base...")
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search parameters (optimized for speed)
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 5}  # Reduced from 10 for faster search
        }
        
        # Perform vector search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="vector",  # Adjust field name if different
            param=search_params,
            limit=TOP_K,
            output_fields=["text", "source_type", "filename", "page_number"]  # Include metadata
        )
        
        # Extract retrieved chunks with metadata
        retrieved_chunks = []
        for hits in results:
            for i, hit in enumerate(hits):
                chunk_text = hit.entity.get("text", "")
                if chunk_text:
                    # Get metadata if available
                    source_type = hit.entity.get("source_type", "unknown")
                    filename = hit.entity.get("filename", "N/A")
                    page_number = hit.entity.get("page_number", "N/A")
                    
                    retrieved_chunks.append(chunk_text)
        
        print(f"  ✅ Retrieved {len(retrieved_chunks)} chunks")
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
        
        # Connect to TTS if not already connected
        if not self.tts_connected:
            if not self.tts.connect():
                print("❌ Failed to connect to TTS service")
                return
            self.tts_connected = True

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
            # Reduced delay for faster response
            time.sleep(0.3)
            
            # Save to history
            self.history.append({
                "user": user_query,
                "assistant": full_response_text,
                "timestamp": time.time()
            })
            self.save_history()
            
        except Exception as e:
            print(f"\n❌ Error during generation/TTS: {e}")
        # Don't close TTS - keep it open for next query
    
    async def run_pipeline(self):
        """Run the complete voice-to-voice RAG pipeline with PDF upload option"""
        
        print("\n" + "="*50)
        print("🎯 VOICE-TO-VOICE RAG PIPELINE")
        print("="*50)
        print("Part 1: Deepgram (Speech-to-Text)")
        print("Part 1.5: Milvus RAG (Retrieval)")
        print("Part 2: Gemini 2.0 Flash (LLM)")
        print("Part 3: Deepgram (Text-to-Speech)")
        print("="*50 + "\n")
        
        # Main menu loop
        while True:
            print("\n" + "="*50)
            print("📋 MAIN MENU")
            print("="*50)
            print("1. 📤 Upload PDF to Knowledge Base")
            print("2. 🎤 Start Voice Chat")
            print("3. 🚪 Exit")
            print("="*50)
            
            choice = input("\nSelect an option (1-3): ").strip()
            
            if choice == "1":
                # PDF Upload
                pdf_path = select_pdf_file()
                if pdf_path:
                    self.upload_and_store_pdf(pdf_path)
                else:
                    print("\n❌ PDF upload cancelled\n")
                    
            elif choice == "2":
                # Voice Chat Loop
                print("\n" + "="*50)
                print("🎤 VOICE CHAT MODE")
                print("="*50)
                print("Say 'exit' or 'stop' to return to main menu\n")
                
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
                        print("\n👋 Returning to main menu...\n")
                        break
                    
                    # Part 1.5: Retrieve relevant context from Milvus
                    context_chunks = self.retrieve_context(user_query)
                    
                    # Part 2 & 3: Generate response and stream TTS
                    await self.generate_and_stream_response(user_query, context_chunks)
                    
                    print("\n✨ Turn complete!")
                
                # Close TTS connection when exiting voice chat
                if self.tts_connected:
                    self.tts.close()
                    self.tts_connected = False
                    
            elif choice == "3":
                # Exit
                print("\n👋 Goodbye!\n")
                break
                
            else:
                print("\n❌ Invalid choice. Please select 1, 2, or 3.\n")
async def main():
    pipeline = VoicePipeline()
    await pipeline.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
