from deepgram import DeepgramClient
from deepgram.extensions.types.sockets import (
    AgentV1SettingsMessage, AgentV1Agent, AgentV1AudioConfig,
    AgentV1AudioInput, AgentV1Listen, AgentV1ListenProvider,
    AgentV1Think, AgentV1OpenAiThinkProvider, AgentV1SpeakProviderConfig,
    AgentV1DeepgramSpeakProvider
)

client = DeepgramClient(api_key="Deepgram_api_key")

with client.agent.v1.connect() as agent:
    settings = AgentV1SettingsMessage(
        audio=AgentV1AudioConfig(
            input=AgentV1AudioInput(encoding="linear16", sample_rate=44100)
        ),
        agent=AgentV1Agent(
            listen=AgentV1Listen(
                provider=AgentV1ListenProvider(type="deepgram", model="nova-3")
            ),
            think=AgentV1Think(
                provider=AgentV1OpenAiThinkProvider(
                    type="open_ai", model="gpt-4o-mini"
                )
            ),
            speak=AgentV1SpeakProviderConfig(
                provider=AgentV1DeepgramSpeakProvider(
                    type="deepgram", model="aura-2-asteria-en"
                )
            )
        )
    )

    agent.send_settings(settings)
    agent.start_listening()
