from typing import TypedDict, List, Literal

class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatML(TypedDict):
    messages: List[Message]
