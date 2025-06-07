from langchain_core.vectorstores import InMemoryVectorStore
from openai import OpenAI


class RAG:
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        instructions_path: str,
        llm_model="gpt-4.1-nano",
    ):
        self.client = OpenAI()
        self.instructions = self._load_instructions(instructions_path)
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.response = None

    def _build_context(self, query: str) -> str:
        results = self.vector_store.similarity_search(query)
        return "\n\n".join([res for res in results])

    def _load_instructions(self, instructions_path: str) -> str:
        instructions = None
        try:
            with open(instructions_path, "r", encoding="utf-8") as f:
                instructions = f.read()
        except Exception as e:
            raise e
        return instructions

    def get_response(self, query: str):
        context = self._build_context(query)
        response = self.client.responses.create(
            model=self.llm_model,
            instructions=self.instructions,
            input=f"Contexto: {context}\n\nPergunta: {query}",
        )
        return response


if __name__ is "__main__":
    pass
