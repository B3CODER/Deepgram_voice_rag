"""
Voice-to-Voice Pipeline
========================
Part 1: Deepgram (Speech-to-Text) - Microphone input
Part 2: Gemini 2.5 Flash (LLM) - Generate response
Part 3: Speechmatics (Text-to-Speech) - Audio output

Requirements:
- deepgram-sdk==3.7.0 (for LiveTranscriptionEvents support)
- speechmatics-flow
- google-generativeai
- pyaudio
"""

import asyncio
import os
import time
import pyaudio
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
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


class VoicePipeline:
    def __init__(self):
        self.deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        self.transcribed_text = ""
        self.final_transcript = ""
        self.is_listening = True
        self.silence_timeout = 2.0  # seconds of silence before processing
        self.last_transcript_time = None
        
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
    
    def generate_response(self, user_query: str) -> str:
        """Part 2: Use Gemini 2.0 Flash to generate response"""
        
        if not user_query:
            return "I didn't catch that. Could you please repeat?"
        
        print(f"\n🤔 Thinking...")
        
        # System prompt - customize as needed
        system_prompt = """You are a helpful voice assistant. 
        Keep your responses concise and conversational since they will be spoken aloud.
        Avoid using markdown, bullet points, or special formatting."""
        
        # Generate response using Gemini
        response = self.gemini_model.generate_content(
            f"{system_prompt}\n\nUser Query: {user_query}"
        )
        
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
    
    async def run_pipeline(self):
        """Run the complete voice-to-voice pipeline"""
        
        print("\n" + "="*50)
        print("🎯 VOICE-TO-VOICE PIPELINE")
        print("="*50)
        print("Part 1: Deepgram (Speech-to-Text)")
        print("Part 2: Gemini 2.0 Flash (LLM)")
        print("Part 3: Speechmatics (Text-to-Speech)")
        print("="*50 + "\n")
        
        # Part 1: Speech to Text (synchronous)
        user_query = self.speech_to_text()
        
        if not user_query:
            print("No speech detected. Exiting.")
            return
        
        # Part 2: Generate response with Gemini
        response_text = self.generate_response(user_query)
        
        # Part 3: Text to Speech
        await self.text_to_speech(response_text)
        
        print("\n✨ Pipeline complete!")


async def main():
    pipeline = VoicePipeline()
    await pipeline.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
