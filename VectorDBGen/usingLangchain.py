import os
import toml
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import time

# LangChain imports
from langchain.document_loaders import DirectoryLoader, BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.retrievers import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TOMLLoader(BaseLoader):
    """Custom LangChain document loader for TOML files."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """Load and parse a TOML file into LangChain documents."""
        documents = []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                toml_content = toml.load(f)
            
            # Extract text content with structure preservation
            text_chunks = self._extract_structured_content(toml_content)
            
            for chunk in text_chunks:
                doc = Document(
                    page_content=chunk['text'],
                    metadata={
                        'source': self.file_path,
                        'file_name': Path(self.file_path).name,
                        'section': chunk['section'],
                        'key': chunk['key'],
                        'value': chunk['value'],
                        'doc_type': 'toml',
                        'timestamp': time.time()
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error loading TOML file {self.file_path}: {str(e)}")
        
        return documents
    
    def _extract_structured_content(self, toml_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract meaningful text content from TOML structure."""
        text_chunks = []
        
        def extract_recursive(data, prefix="", section_path=""):
            """Recursively extract text from nested TOML structure"""
            if isinstance(data, dict):
                for key, value in data.items():
                    current_section = f"{section_path}.{key}" if section_path else key
                    current_prefix = f"{prefix}[{key}] " if prefix else f"[{key}] "
                    
                    if isinstance(value, (dict, list)):
                        extract_recursive(value, current_prefix, current_section)
                    else:
                        # Convert value to string and create chunk
                        text_value = str(value)
                        if text_value.strip():  # Only add non-empty content
                            chunk_text = f"{current_prefix}{text_value}"
                            text_chunks.append({
                                'text': chunk_text,
                                'section': current_section,
                                'key': key,
                                'value': text_value
                            })
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    list_prefix = f"{prefix}[{i}] "
                    current_section = f"{section_path}[{i}]"
                    extract_recursive(item, list_prefix, current_section)
        
        extract_recursive(toml_content)
        return text_chunks

class TOMLToVectorDB:
    """LangChain-powered TOML to Vector Database converter."""
    
    def __init__(
        self, 
        azure_endpoint: str,
        api_key: str,
        deployment_name: str = "text-embeddings-002",
        api_version: str = "2023-05-15",
        vector_db_path: str = "./chroma_db",
        collection_name: str = "toml_knowledge_base",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the TOML to Vector DB converter with LangChain.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            deployment_name: Deployment name for text-embeddings-002
            api_version: API version
            vector_db_path: Path to store ChromaDB
            collection_name: Name for the vector collection
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.azure_endpoint = azure_endpoint.rstrip('/')
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        
        # Initialize LangChain components
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            azure_deployment=deployment_name,
            api_version=api_version,
            model=deployment_name
        )
        
        # Text splitter for additional chunking if needed
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "[", "]", ".", " ", ""],
            keep_separator=True
        )
        
        # Vector store (will be initialized after processing)
        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[VectorStoreRetriever] = None
    
    def load_toml_documents(self, directory: str) -> List[Document]:
        """
        Load all TOML documents from a directory using LangChain.
        
        Args:
            directory: Directory containing TOML files
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Loading TOML documents from {directory}")
        
        # Use DirectoryLoader with custom TOML loader
        loader = DirectoryLoader(
            directory,
            glob="**/*.toml",
            loader_cls=TOMLLoader,
            show_progress=True,
            use_multithreading=True
        )
        
        try:
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} document chunks from TOML files")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return []
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents with additional text splitting if needed.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Processed documents ready for embedding
        """
        logger.info("Processing documents for optimal chunking...")
        
        # Filter out very short documents and apply additional splitting if needed
        processed_docs = []
        
        for doc in documents:
            # Skip very short content
            if len(doc.page_content.strip()) < 10:
                continue
            
            # Apply text splitting if content is too long
            if len(doc.page_content) > self.text_splitter._chunk_size:
                chunks = self.text_splitter.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    # Update metadata to include chunk info
                    chunk.metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_length': len(doc.page_content)
                    })
                processed_docs.extend(chunks)
            else:
                processed_docs.append(doc)
        
        logger.info(f"Processed into {len(processed_docs)} final document chunks")
        return processed_docs
    
    async def create_vector_store(self, directory: str) -> None:
        """
        Create vector store from TOML files in directory.
        
        Args:
            directory: Directory containing TOML files
        """
        logger.info("Starting TOML to Vector DB conversion with LangChain...")
        
        # Load documents
        documents = self.load_toml_documents(directory)
        if not documents:
            logger.warning("No documents loaded")
            return
        
        # Process documents
        processed_docs = self.process_documents(documents)
        if not processed_docs:
            logger.warning("No documents to process")
            return
        
        # Create vector store
        logger.info("Creating vector store with embeddings...")
        try:
            self.vectorstore = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: Chroma.from_documents(
                    documents=processed_docs,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    persist_directory=self.vector_db_path
                )
            )
            
            # Persist the vector store
            self.vectorstore.persist()
            
            # Initialize retriever
            self.retriever = VectorStoreRetriever(
                vectorstore=self.vectorstore,
                search_kwargs={
                    "k": 5,
                    "fetch_k": 20
                }
            )
            
            logger.info("Successfully created vector store!")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
    
    def load_existing_vector_store(self) -> bool:
        """
        Load an existing vector store from disk.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.vector_db_path
            )
            
            # Check if collection has content
            if self.vectorstore._collection.count() == 0:
                logger.warning("Vector store exists but is empty")
                return False
            
            self.retriever = VectorStoreRetriever(
                vectorstore=self.vectorstore,
                search_kwargs={
                    "k": 5,
                    "fetch_k": 20
                }
            )
            
            logger.info(f"Loaded existing vector store with {self.vectorstore._collection.count()} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading existing vector store: {str(e)}")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        try:
            if filter_metadata:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vectorstore.similarity_search(query=query, k=k)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_scores(
        self, 
        query: str, 
        k: int = 5
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query=query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with scores: {str(e)}")
            return []
    
    def get_retriever(self, search_type: str = "similarity", **kwargs) -> Optional[VectorStoreRetriever]:
        """
        Get a configured retriever for the vector store.
        
        Args:
            search_type: Type of search ("similarity", "similarity_score_threshold", "mmr")
            **kwargs: Additional arguments for retriever configuration
            
        Returns:
            Configured retriever or None
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return None
        
        try:
            retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=kwargs or {"k": 5}
            )
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection."""
        if not self.vectorstore:
            return {"error": "Vector store not initialized"}
        
        try:
            count = self.vectorstore._collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'database_path': self.vector_db_path,
                'embedding_model': self.deployment_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store.
        
        Args:
            documents: List of documents to add
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return
        
        try:
            processed_docs = self.process_documents(documents)
            self.vectorstore.add_documents(processed_docs)
            self.vectorstore.persist()
            logger.info(f"Added {len(processed_docs)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")

# Example usage and utility functions
async def main():
    """Example usage of the TOML to Vector DB converter."""
    
    # Configuration
    AZURE_ENDPOINT = "https://your-resource-name.openai.azure.com"
    API_KEY = "your-api-key-here"
    DEPLOYMENT_NAME = "text-embeddings-002"
    TOML_DIRECTORY = "./knowledge_base"  # Directory containing your TOML files
    
    # Initialize converter
    converter = TOMLToVectorDB(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        deployment_name=DEPLOYMENT_NAME,
        vector_db_path="./vector_db",
        collection_name="toml_knowledge_base",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Try to load existing vector store, otherwise create new one
    if not converter.load_existing_vector_store():
        logger.info("Creating new vector store...")
        await converter.create_vector_store(TOML_DIRECTORY)
    
    # Get collection statistics
    stats = converter.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Example searches
    if stats.get('total_documents', 0) > 0:
        # Basic similarity search
        query = "your search query here"
        results = converter.similarity_search(query, k=3)
        print(f"\nSimilarity search results for '{query}':")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content[:200]}...")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Section: {doc.metadata.get('section', 'Unknown')}")
        
        # Search with scores
        scored_results = converter.similarity_search_with_scores(query, k=3)
        print(f"\nSearch results with relevance scores:")
        for doc, score in scored_results:
            print(f"Score: {score:.4f}")
            print(f"Content: {doc.page_content[:150]}...")
            print(f"File: {doc.metadata.get('file_name', 'Unknown')}")
            print("---")
        
        # Filtered search example
        filter_criteria = {"file_name": "specific_file.toml"}  # Adjust as needed
        filtered_results = converter.similarity_search(
            query, 
            k=3, 
            filter_metadata=filter_criteria
        )
        print(f"\nFiltered search results: {len(filtered_results)} documents")

if __name__ == "__main__":
    asyncio.run(main())