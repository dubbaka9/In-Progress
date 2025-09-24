import os
import toml
import json
import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import logging
from pathlib import Path
import hashlib
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TOMLToVectorDB:
    def __init__(
        self, 
        azure_endpoint: str,
        api_key: str,
        deployment_name: str = "text-embeddings-002",
        api_version: str = "2023-05-15",
        vector_db_path: str = "./chroma_db",
        collection_name: str = "toml_knowledge_base"
    ):
        """
        Initialize the TOML to Vector DB converter.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            deployment_name: Deployment name for text-embeddings-002
            api_version: API version
            vector_db_path: Path to store ChromaDB
            collection_name: Name for the vector collection
        """
        self.azure_endpoint = azure_endpoint.rstrip('/')
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Knowledge base from TOML files"}
        )
        
        # Embedding endpoint
        self.embedding_url = f"{self.azure_endpoint}/openai/deployments/{self.deployment_name}/embeddings"
    
    def read_toml_files(self, directory: str) -> List[Dict[str, Any]]:
        """
        Read all TOML files from a directory and return their contents.
        
        Args:
            directory: Path to directory containing TOML files
            
        Returns:
            List of dictionaries containing file path and parsed content
        """
        toml_data = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Directory {directory} does not exist")
            return toml_data
        
        for toml_file in directory_path.rglob("*.toml"):
            try:
                with open(toml_file, 'r', encoding='utf-8') as f:
                    content = toml.load(f)
                    toml_data.append({
                        'file_path': str(toml_file),
                        'content': content,
                        'file_name': toml_file.name
                    })
                logger.info(f"Successfully read {toml_file}")
            except Exception as e:
                logger.error(f"Error reading {toml_file}: {str(e)}")
        
        logger.info(f"Read {len(toml_data)} TOML files")
        return toml_data
    
    def extract_text_content(self, toml_content: Dict[str, Any], file_path: str) -> List[Dict[str, Any]]:
        """
        Extract meaningful text content from TOML structure for embedding.
        
        Args:
            toml_content: Parsed TOML content
            file_path: Original file path
            
        Returns:
            List of text chunks with metadata
        """
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
                                'value': text_value,
                                'file_path': file_path
                            })
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    list_prefix = f"{prefix}[{i}] "
                    current_section = f"{section_path}[{i}]"
                    extract_recursive(item, list_prefix, current_section)
        
        extract_recursive(toml_content)
        return text_chunks
    
    async def get_embeddings(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Get embeddings for a list of texts using Azure OpenAI.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                payload = {
                    "input": batch,
                    "model": self.deployment_name
                }
                
                params = {"api-version": self.api_version}
                
                try:
                    async with session.post(
                        self.embedding_url,
                        headers=headers,
                        json=payload,
                        params=params
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            batch_embeddings = [item["embedding"] for item in result["data"]]
                            all_embeddings.extend(batch_embeddings)
                            logger.info(f"Generated embeddings for batch {i//batch_size + 1}")
                        else:
                            error_text = await response.text()
                            logger.error(f"API request failed with status {response.status}: {error_text}")
                            # Add placeholder embeddings to maintain alignment
                            all_embeddings.extend([[0.0] * 1536] * len(batch))
                    
                    # Rate limiting - wait between batches
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {str(e)}")
                    # Add placeholder embeddings to maintain alignment
                    all_embeddings.extend([[0.0] * 1536] * len(batch))
        
        return all_embeddings
    
    def generate_chunk_id(self, file_path: str, section: str, text: str) -> str:
        """Generate a unique ID for each text chunk."""
        content = f"{file_path}:{section}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def process_and_store(self, directory: str, batch_size: int = 10):
        """
        Process all TOML files in directory and store in vector database.
        
        Args:
            directory: Directory containing TOML files
            batch_size: Batch size for embedding generation
        """
        logger.info("Starting TOML to Vector DB conversion...")
        
        # Read all TOML files
        toml_files = self.read_toml_files(directory)
        if not toml_files:
            logger.warning("No TOML files found")
            return
        
        # Extract text chunks from all files
        all_chunks = []
        for file_data in toml_files:
            chunks = self.extract_text_content(file_data['content'], file_data['file_path'])
            all_chunks.extend(chunks)
        
        logger.info(f"Extracted {len(all_chunks)} text chunks")
        
        if not all_chunks:
            logger.warning("No text content found in TOML files")
            return
        
        # Prepare texts for embedding
        texts = [chunk['text'] for chunk in all_chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = await self.get_embeddings(texts, batch_size)
        
        if len(embeddings) != len(all_chunks):
            logger.error("Mismatch between number of chunks and embeddings")
            return
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents = []
        embedding_vectors = []
        
        for chunk, embedding in zip(all_chunks, embeddings):
            # Skip chunks with placeholder embeddings
            if not any(val != 0.0 for val in embedding):
                continue
                
            chunk_id = self.generate_chunk_id(chunk['file_path'], chunk['section'], chunk['text'])
            ids.append(chunk_id)
            
            metadata = {
                'file_path': chunk['file_path'],
                'section': chunk['section'],
                'key': chunk['key'],
                'value': chunk['value'],
                'file_name': Path(chunk['file_path']).name,
                'timestamp': time.time()
            }
            metadatas.append(metadata)
            documents.append(chunk['text'])
            embedding_vectors.append(embedding)
        
        # Store in ChromaDB
        if ids:
            logger.info(f"Storing {len(ids)} chunks in vector database...")
            
            # ChromaDB has a limit on batch size, so we'll process in smaller batches
            storage_batch_size = 100
            for i in range(0, len(ids), storage_batch_size):
                batch_end = min(i + storage_batch_size, len(ids))
                
                try:
                    self.collection.add(
                        ids=ids[i:batch_end],
                        documents=documents[i:batch_end],
                        metadatas=metadatas[i:batch_end],
                        embeddings=embedding_vectors[i:batch_end]
                    )
                    logger.info(f"Stored batch {i//storage_batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error storing batch {i//storage_batch_size + 1}: {str(e)}")
            
            logger.info("Successfully completed TOML to Vector DB conversion!")
        else:
            logger.warning("No valid chunks to store")
    
    def search_similar(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar content in the vector database.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results with documents and metadata
        """
        try:
            # Generate embedding for query
            loop = asyncio.get_event_loop()
            query_embedding = loop.run_until_complete(self.get_embeddings([query]))[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            return results
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return {}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection."""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'database_path': self.vector_db_path
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

# Example usage
async def main():
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
        collection_name="toml_knowledge_base"
    )
    
    # Process and store TOML files
    await converter.process_and_store(TOML_DIRECTORY, batch_size=5)
    
    # Get collection statistics
    stats = converter.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Example search
    if stats.get('total_chunks', 0) > 0:
        results = converter.search_similar("your search query here", n_results=3)
        print(f"Search results: {results}")

if __name__ == "__main__":
    asyncio.run(main())