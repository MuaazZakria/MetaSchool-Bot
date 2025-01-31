import os
import base64
import requests
from typing import List, Dict, Optional, Callable
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


class GitHubFileLoader:
    def __init__(self, repo: str, access_token: str, branch: str = "main", file_filter: Optional[Callable[[str], bool]] = None):
        """Initializes the GitHubFileLoader with repository details and optional file filter."""
        self.repo = repo
        self.branch = branch
        self.file_filter = file_filter
        self.headers = {"Authorization": f"token {access_token}"}
        self.github_api_url = "https://api.github.com"

    def get_file_paths(self) -> List[Dict]:
        """Fetch all file paths from the repository."""
        url = f"{self.github_api_url}/repos/{self.repo}/git/trees/{self.branch}?recursive=1"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        all_files = response.json().get("tree", [])
        return [
            file for file in all_files
            if file["type"] == "blob" and (not self.file_filter or self.file_filter(file["path"]))
        ]

    def get_file_content_by_path(self, path: str) -> str:
        """Fetch the content of a specific file by its path."""
        from urllib.parse import quote
        encoded_path = quote(path)
        url = f"{self.github_api_url}/repos/{self.repo}/contents/{encoded_path}?ref={self.branch}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        content_encoded = response.json().get("content", "")
        return base64.b64decode(content_encoded).decode("utf-8") if content_encoded else ""

    def load_documents(self) -> List[Document]:
        """Load all documents as LangChain Document objects."""
        documents = []
        files = self.get_file_paths()
        for file in files:
            try:
                content = self.get_file_content_by_path(file["path"])
                if not content:
                    continue
                metadata = {
                    "path": file["path"],
                    "sha": file["sha"],
                    "source": f"https://github.com/{self.repo}/blob/{self.branch}/{file['path']}"
                }
                documents.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                print(f"Error loading file {file['path']}: {e}")
        return documents


def process_documents(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 100) -> List[Dict]:
    """Split documents into chunks and associate metadata."""
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks_with_metadata = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunks_with_metadata.append({
                "content": chunk,
                "metadata": doc.metadata
            })
    return chunks_with_metadata


def create_pinecone_index(index_name: str, dimension: int = 1536) -> PineconeVectorStore:
    """Create a Pinecone index and return a vector store."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(index_name)
    return index


def embed_and_store_documents(chunks_with_metadata: List[Dict], vectorstore: PineconeVectorStore) -> None:
    """Embed documents and store them in Pinecone."""
    documents = [Document(
        page_content=chunk["content"],
        metadata=chunk["metadata"]
    ) for chunk in chunks_with_metadata]
    vectorstore.add_documents(documents)


def main():
    load_dotenv()  # Load environment variables

    # Replace with your GitHub repository and token
    repo_name = "0xmetaschool/Learning-Projects"
    access_token = "github-access-token"

    # Optional file filter: e.g., process only Markdown files
    def file_filter(path: str) -> bool:
        return path.endswith(".md")

    # Initialize GitHubFileLoader
    loader = GitHubFileLoader(repo=repo_name, access_token=access_token, file_filter=file_filter)
    documents = loader.load_documents()

    # Process documents into chunks
    chunks_with_metadata = process_documents(documents)

    # Initialize the Pinecone index
    index_name = "metaschool-new-index-2"
    index = create_pinecone_index(index_name)

    # Create PineconeVectorStore with OpenAI embeddings
    model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPEN_AI_KEY"))
    vectorstore = PineconeVectorStore(index=index, embedding=model, namespace="default_namespace")

    # Embed and store documents in Pinecone
    embed_and_store_documents(chunks_with_metadata, vectorstore)


if __name__ == "__main__":
    main()
