from transformers import pipeline
import transformers
import logging
import os
from config import *
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CHANNEL_MAP = {
    # irritation
    "anger":         {"irritation": 1.0},
    "annoyance":     {"irritation": 0.8},
    "disapproval":   {"irritation": 0.6},
    "disgust":       {"irritation": 0.7},
    "frustration":   {"irritation": 0.9},

    # warmth
    "love":          {"warmth": 1.0},
    "caring":        {"warmth": 0.9},
    "gratitude":     {"warmth": 0.7},
    "admiration":    {"warmth": 0.6},
    "approval":      {"warmth": 0.4},
    "relief":        {"warmth": 0.3},

    # amusement
    "amusement":     {"amusement": 1.0},
    "joy":           {"amusement": 0.6},
    "excitement":    {"amusement": 0.5},
    "pride":         {"amusement": 0.4},

    # melancholy
    "sadness":       {"melancholy": 1.0},
    "grief":         {"melancholy": 1.0},
    "disappointment":{"melancholy": 0.7},
    "remorse":       {"melancholy": 0.6},
    "embarrassment": {"melancholy": 0.4},

    # unease
    "fear":          {"unease": 1.0},
    "nervousness":   {"unease": 0.8},
    "confusion":     {"unease": 0.5},
    "surprise":      {"unease": 0.3},

    # interest
    "curiosity":     {"interest": 1.0},
    "realization":   {"interest": 0.7},
    "desire":        {"interest": 0.6},
    "optimism":      {"interest": 0.5},

    # neutral contributes nothing
    "neutral":       {},
}

DECAY_RATES = {
    "irritation":  0.15,  # fades medium — lingers but doesn't last forever
    "warmth":      0.03,  # fades slow — hard to earn, hard to lose
    "amusement":   0.25,  # fades fast — in the moment only
    "melancholy":  0.02,  # fades very slow — sticks around for a long time
    "unease":      0.10,  # fades medium
    "interest":    0.20,  # fades fast — needs constant novelty to maintain
}

PULL_STRENGTH = 0.4  # tune this up/down to make emotions more/less reactive

BASELINE_STATE = {
    "irritation":  0.2,
    "warmth":      0.05,
    "amusement":   0.0,
    "melancholy":  0.1,
    "unease":      0.0,
    "interest":    0.15,
}


class Emotions():
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            device=0,
            model_kwargs={"ignore_mismatched_sizes": True},
            verbose=False
        )
        # State is reduced from 28 emotions to 6 core dimensions for easier management
        self.state = dict(BASELINE_STATE)

    def react(self, text):
        results = self.classifier(text)
        
        for emotion in results[0]:
            if emotion['score'] < 0.05:  # noise threshold
                continue
            label = emotion['label'].lower()
            if label in CHANNEL_MAP:
                for channel, weight in CHANNEL_MAP[label].items():
                    self.state[channel] += weight * emotion['score'] * PULL_STRENGTH

        # Clamp after all contributions are summed
        for channel in self.state:
            self.state[channel] = min(max(self.state[channel], 0.0), 1.0)

        # Decay toward baseline
        self.decay()
    
    def decay(self):
        for channel in self.state:
            self.state[channel] += DECAY_RATES[channel] * (BASELINE_STATE[channel] - self.state[channel])
    
    def value_to_word(self, value):
        if value < 0.05:  return "absent"
        if value < 0.15:  return "barely present"
        if value < 0.25:  return "faint"
        if value < 0.35:  return "mild"
        if value < 0.45:  return "noticeable"
        if value < 0.55:  return "moderate"
        if value < 0.65:  return "pronounced"
        if value < 0.75:  return "strong"
        if value < 0.85:  return "intense"
        if value < 0.95:  return "overwhelming"
        return "all-consuming"

    def as_prompt(self):
        lines = [f"- {channel}: {self.value_to_word(value)}" 
                for channel, value in self.state.items()]
        body = "\n".join(lines)
        return f"Your current emotional state:\n{body}\n\nLet this color your tone and word choice — never reference these values directly."
