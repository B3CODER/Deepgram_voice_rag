import asyncio
import os
from dotenv import load_dotenv

from speechmatics.tts import AsyncClient, Voice, OutputFormat

# Load the .env file
load_dotenv()

API_KEY = os.getenv("SPEECHMATICS_API_KEY")

async def main():
    # Pass API key into AsyncClient
    async with AsyncClient(api_key=API_KEY) as client:
        async with await client.generate(
            text="Welcome to the future of voice AI!",
            voice=Voice.SARAH,
            output_format=OutputFormat.WAV_16000
        ) as response:
            audio = b''.join([chunk async for chunk in response.content.iter_chunked(1024)])
            with open("output.wav", "wb") as f:
                f.write(audio)

if __name__ == "__main__":
    asyncio.run(main())
