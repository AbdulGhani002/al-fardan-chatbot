"""Voice-call agent.

Wires a phone number (via Twilio Voice webhooks) into the same
retrieval + RAG pipeline the chat widget uses, so the bot answers
phone calls with the exact same Safiya persona + KB + Sharia rulings.

Stack:
  Twilio / SIP  →  /voice/incoming  →  Twilio transcribes the caller
                →  /voice/respond   →  RAG answers  →  TwiML <Say>

First-iteration uses Twilio's built-in STT + TTS — zero external
audio dependencies on our side. Swappable to Whisper + Piper for
higher quality once the basic call flow is proven.
"""
