from .base import BaseProcessor, ProcessorRegistry
from custom_types import ChatTemplate, ChatMessage


@ProcessorRegistry.register
class EmotionProcessor(BaseProcessor):
    dataset_id = "dair-ai/emotion"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_list = ["sadness","joy","love","anger","fear","surprise"]
        
    def to_chat_template(self, example) -> ChatTemplate:
        prompt = f"Below is a piece of text. Classify it into one of: {', '.join(self.labels_list)}.\n\n\"{example['text']}\""

        if self.with_assistant:
            response = f"The emotion in the above text is: {self.labels_list[example['label']]}"
            return ChatTemplate(
                messages=[
                    ChatMessage(role="user",content=prompt),
                    ChatMessage(role="assistant", content=response)
                ]
            )
        else:
            return ChatTemplate(
                messages=[
                    ChatMessage(role="user",content=prompt),
                ]
            )
