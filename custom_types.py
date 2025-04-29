from typing import TypedDict, List, Literal


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str

class ChatTemplate(TypedDict):
    messages: List[ChatMessage]
