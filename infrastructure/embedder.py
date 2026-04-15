import infrastructure.config as config

class Embedder:
    """Responsible for generating and caching embeddings for text inputs."""
    def __init__(self, client):
        self.client = client
        self.embedding_cache = {}

    def get_embedding(self, text):
        """Get the embedding for a given text, using caching to avoid redundant API calls."""
        text = text.replace("\n", " ")
        if text in self.embedding_cache:
                return self.embedding_cache[text]
        response = self.client.embeddings.create(
                input=[text],
                model=config.EMBED_MODEL
        )
        self.embedding_cache[text] = response.data[0].embedding
        return self.embedding_cache[text]
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache = {}