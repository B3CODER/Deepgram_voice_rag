"""
Voice-to-Voice RAG Pipeline
============================
Part 1: Deepgram (Speech-to-Text) - Microphone input
Part 1.5: Milvus RAG (Retrieval) - Search relevant documents
Part 2: Gemini 2.5 Flash (LLM) - Generate response with context
Part 3: Speechmatics (Text-to-Speech) - Audio output

Requirements:
- deepgram-sdk==3.7.0 (for LiveTranscriptionEvents support)
- speechmatics-flow
- google-generativeai
- pyaudio
- pymilvus
- sentence-transformers
"""

import asyncio
import os
import time
import pyaudio
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load environment variables
load_dotenv()

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Audio settings for microphone
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Milvus Configuration
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_DB = "DEV_DB"
MILVUS_COLLECTION = "company_abc123"
TOP_K = 5  # Number of chunks to retrieve


class VoicePipeline:
    def __init__(self):
        self.deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
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
        print(f"✅ Connected to collection: {MILVUS_COLLECTION}\n")
        
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
            model="nova-2",
            language="en",
            smart_format=True,
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            interim_results=True,
            endpointing=300,  # milliseconds of silence for endpoint detection
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
        
        print(f"\n🔍 Searching knowledge base...")
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query).tolist()
        
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
    
    def generate_response(self, user_query: str, context_chunks: list = None) -> str:
        """Part 2: Use Gemini 2.5 Flash to generate response with RAG context"""
        
        if not user_query:
            return "I didn't catch that. Could you please repeat?"
        
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
        
        # Generate response using Gemini
        response = self.gemini_model.generate_content(full_prompt)
        
        response_text = response.text
        print(f"🤖 Assistant: {response_text}")
        
        return response_text
    
    async def text_to_speech(self, text: str, output_file: str = "response.wav"):
        """Part 3: Use Speechmatics to convert text to speech"""
        # Import here to avoid import errors if package not installed
        from speechmatics.tts import AsyncClient as SpeechmaticsClient, Voice, OutputFormat
        
        print(f"\n🔊 Converting to speech...")
        
        async with SpeechmaticsClient(api_key=SPEECHMATICS_API_KEY) as client:
            async with await client.generate(
                text=text,
                voice=Voice.SARAH,
                output_format=OutputFormat.WAV_16000
            ) as response:
                audio = b''.join([chunk async for chunk in response.content.iter_chunked(1024)])
                with open(output_file, "wb") as f:
                    f.write(audio)
        
        print(f"✅ Audio saved to: {output_file}")
        return output_file
    
    def play_audio(self, audio_file: str):
        """Play the audio file using PyAudio"""
        import wave
        
        print(f"\n🔈 Playing audio...")
        
        # Open the wave file
        wf = wave.open(audio_file, 'rb')
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        # Open stream
        stream = audio.open(
            format=audio.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        
        # Read and play audio in chunks
        chunk_size = 1024
        data = wf.readframes(chunk_size)
        
        while data:
            stream.write(data)
            data = wf.readframes(chunk_size)
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        audio.terminate()
        wf.close()
        
        print("🔈 Audio playback complete!")
    
    async def run_pipeline(self):
        """Run the complete voice-to-voice RAG pipeline"""
        
        print("\n" + "="*50)
        print("🎯 VOICE-TO-VOICE RAG PIPELINE")
        print("="*50)
        print("Part 1: Deepgram (Speech-to-Text)")
        print("Part 1.5: Milvus RAG (Retrieval)")
        print("Part 2: Gemini 2.5 Flash (LLM)")
        print("Part 3: Speechmatics (Text-to-Speech)")
        print("="*50 + "\n")
        
        # Part 1: Speech to Text (synchronous)
        user_query = self.speech_to_text()
        
        if not user_query:
            print("No speech detected. Exiting.")
            return
        
        # Part 1.5: Retrieve relevant context from Milvus
        context_chunks = self.retrieve_context(user_query)
        
        # Part 2: Generate response with Gemini using RAG context
        response_text = self.generate_response(user_query, context_chunks)
        
        # Part 3: Text to Speech
        audio_file = await self.text_to_speech(response_text)
        
        # Part 4: Play the response audio
        self.play_audio(audio_file)
        
        print("\n✨ Pipeline complete!")


async def main():
    pipeline = VoicePipeline()
    await pipeline.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
