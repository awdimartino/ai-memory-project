"""Local RoBERTa GoEmotions classifier (the emotion "sense organ").

Wraps a HuggingFace text-classification pipeline that scores a message across
the 28 GoEmotions labels. Infrastructure-only: it turns text into raw (label,
score) pairs; the 28->6 channel mapping and mood dynamics live in the core
EmotionManager (mirrors the embedder / memory-manager split, so the mood logic
stays unit-testable without torch).

Runs on **CPU** (`device=-1`): the model is ~125M params, per-message inference
is cheap, and it keeps all 16GB VRAM for the LLMs (the v1 CUDA `device=0` path
isn't available on the AMD 9070XT). Construction loads the model (blocking, and
downloads it on first use), so build it once at startup and off the event loop.
"""
import logging
import os
import threading

logger = logging.getLogger(__name__)

# Keep tokenizers quiet and single-threaded-friendly under the async runtime.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_CACHE: dict[tuple[str, int], "EmotionClassifier"] = {}
_CACHE_LOCK = threading.Lock()


def get_classifier(model_name: str, device: int = -1) -> "EmotionClassifier":
    """A process-wide shared classifier, loading it at most once per (model, device).

    Construction is the expensive part (~seconds of torch/transformers work); inference
    is milliseconds. The app builds one companion and never noticed, but the gold-set
    runner builds a fresh companion PER CASE and so reloaded RoBERTa 138 times per run,
    for no behavioural difference — the pipeline is stateless across `classify` calls.

    Sharing one instance across threads is what makes that safe to do concurrently, so
    `classify` takes a lock (see below). Call the class directly if you genuinely want
    an isolated instance.
    """
    key = (model_name, device)
    with _CACHE_LOCK:                      # so two builds can't both pay the load
        hit = _CACHE.get(key)
        if hit is None:
            hit = _CACHE[key] = EmotionClassifier(model_name, device)
        return hit


class EmotionClassifier:
    def __init__(self, model_name: str, device: int = -1):
        # Import lazily so the module (and the rest of the app) doesn't pay the
        # heavy torch/transformers import unless emotion is actually enabled.
        from transformers import pipeline
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        logger.info("loading emotion classifier %s on %s", model_name,
                    "cpu" if device < 0 else f"cuda:{device}")
        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,          # return all 28 labels with scores
            device=device,
        )
        # One instance may now be shared (see get_classifier), and callers reach
        # `classify` through `asyncio.to_thread`, so two threads can land in the HF
        # pipeline at once. Inference is ~10ms against LLM calls measured in seconds,
        # so serializing it costs nothing measurable and removes the question of
        # whether a given transformers version is re-entrant.
        self._lock = threading.Lock()

    def classify(self, text: str) -> list[tuple[str, float]]:
        """Score `text`; return (label_lowercased, score) pairs, best first.

        Blocking (torch) — call via a threadpool from async code.
        """
        text = (text or "").strip()
        if not text:
            return []
        with self._lock:
            results = self._pipe(text)
        # pipeline returns [[{label, score}, ...]] for a single input.
        row = results[0] if results and isinstance(results[0], list) else results
        pairs = [(r["label"].lower(), float(r["score"])) for r in row]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs
