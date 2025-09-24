from .base import BasePrompter, PrompterRegistry
from src.utils.chat_types import ChatML, Message


@PrompterRegistry.register
class EmotionPrompter(BasePrompter):    
    dataset_id = "dair-ai/emotion"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_list = ["sadness","joy","love","anger","fear","surprise"]
        
    def to_chat_template(self, example) -> ChatML:
        prompt = f"Below is a piece of text. Classify it into one of: {', '.join(self.labels_list)}.\n\n\"{example['text']}\""

        if self.with_assistant:
            response = f"The emotion in the above text is: {self.labels_list[example['label']]}"
            return ChatML(
                messages=[
                    Message(role="user",content=prompt),
                    Message(role="assistant", content=response)
                ]
            )
        else:
            return ChatML(
                messages=[
                    Message(role="user",content=prompt),
                ]
            )
