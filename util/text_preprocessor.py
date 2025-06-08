from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import os


class TextPreprocessor:
    """
    A class for preprocessing text documents, including loading, splitting, and embedding.
    """

    def __init__(self, uploaded_file, embedding_model: str = "text-embedding-3-small"):
        """
        Initializes the TextPreprocessor with a file, and embedding model.

        Args:
            uploaded_file: The uploaded file to process.
            embedding_model (str, optional): The embedding model to use. Defaults to "text-embedding-3-small".
        """
        self.file_path = os.path.join(os.getcwd(), "data", f"temp_{uploaded_file.name}")

        with open(self.file_path, "wb") as f:
            f.write(uploaded_file.read())

        self.docs = None
        self.embedding_model = embedding_model
        self.splits = None
        self.vector_store = None

    def load_file(self) -> List[Document]:
        """
        Loads the file into a list of Document objects.

        Returns:
            List[Document]: A list of Document objects representing the loaded file.
        """
        loader = PyPDFLoader(self.file_path)
        self.docs = loader.load()
        return self.docs

    def split_text(self) -> List[Document]:
        """
        Splits the loaded documents into smaller chunks.

        Raises:
            ValueError: If self.docs is not a valid value (None).

        Returns:
            List[Document]: A list of Document objects representing the split text.
        """
        if not self.docs:
            raise ValueError("self.docs is not a valid value")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splits = splitter.split_documents(self.docs)
        return self.splits

    def load_embeddings(self) -> InMemoryVectorStore:
        """
        Loads the embeddings for the split text into an InMemoryVectorStore.

        Raises:
            ValueError: If self.embedding_model is not a valid value (None).
            ValueError: If self.splits is not a valid value (None).

        Returns:
            InMemoryVectorStore: An InMemoryVectorStore containing the embeddings for the split text.
        """
        if not self.embedding_model:
            raise ValueError("self.embedding_model is not a valid value")
        if not self.splits:
            raise ValueError("self.splits is not a valid value")

        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = InMemoryVectorStore(embeddings)
        self.vector_store.add_documents(self.splits)
        return self.vector_store


if __name__ == "__main__":
    pass
