import voyageai
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions.EmbeddingException import EmbeddingException
from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer

class VoyageEmbeddingEngine(EmbeddingEngine):
    def __init__(self, api_key: str, model: str, dimensions: int, max_tokens: int):
        self.api_key = api_key
        self.model = model
        self.dimensions =    dimensions
        self.max_tokens = max_tokens
        self.tokenizer = self.get_tokenizer()
        self.client = voyageai.AsyncClient(api_key=self.api_key)

    async def embed_text(self, text: str) -> list[float]:
        try:
            result = await self.client.embed(text, model=self.model)
            return result.embeddings
        except Exception as e:
            raise EmbeddingException(f"Error embedding text with Voyage: {e}")

    def get_vector_size(self) -> int:
        return 1024

    def get_tokenizer(self):
        tokenizer = HuggingFaceTokenizer(model="voyageai/" + self.model, max_tokens=self.max_tokens)
        return tokenizer