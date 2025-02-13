import voyageai
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions.EmbeddingException import EmbeddingException

class VoyageEmbeddingEngine(EmbeddingEngine):
    def __init__(self, api_key: str, model: str, dimensions: int, max_tokens: int):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.client = voyageai.Client(api_key=self.api_key)

    async def embed_text(self, text: str) -> list[float]:
        try:
            result = self.client.embed([text], model=self.model)
            return result[0]['embedding']
        except Exception as e:
            raise EmbeddingException(f"Error embedding text with Voyage: {e}")

    def get_vector_size(self) -> int:
        return self.dimensions
