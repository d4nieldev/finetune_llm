"""
This module contains a prompter for every dataset to fine tune the model on.
"""
from .base import PrompterRegistry
from .dair_ai_emotion import EmotionPrompter
from .qpl import *