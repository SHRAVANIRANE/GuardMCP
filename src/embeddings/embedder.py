import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()


class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except OSError:
            self.model = SentenceTransformer(model_name)
        self._cache = {}

    def encode(self, text):
        normalized_text = str(text).strip()
        if normalized_text not in self._cache:
            self._cache[normalized_text] = self.model.encode(normalized_text)
        return self._cache[normalized_text]
