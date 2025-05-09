"""
This module contains a processor for every dataset to fine tune the model on.
"""
from .base import ProcessorRegistry
from .dair_ai_emotion import EmotionProcessor
from .qpl import *