"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
# Removed: from supabase import create_client, Client
from urllib.parse import urlparse
import openai
import chromadb
from chromadb.utils import embedding_functions
from more_itertools import batched # Added for batching ChromaDB inserts

# Load OpenAI API key for contextual embeddings
# Note: Primary vector embeddings are handled by ChromaDB's configured embedding function
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chroma_client(persist_directory: str = "./chroma_db_mcp") -> chromadb.PersistentClient:
    """
    Get a ChromaDB client with the specified persistence directory.
    
    Args:
        persist_directory: Directory where ChromaDB will store its data (defaults to ./chroma_db_mcp)
        
    Returns:
        A ChromaDB PersistentClient
    """
    # Use environment variable if set, otherwise use default
    db_path = os.getenv("CHROMA_DB_PATH", persist_directory)
    
    # Create the directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)
    
    # Return the client
    return chromadb.PersistentClient(db_path)

def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    embedding_model_name: str = "all-MiniLM-L6-v2", # Using a local SentenceTransformer model
    distance_function: str = "cosine",
) -> chromadb.Collection:
    """Get an existing collection or create a new one if it doesn't exist."""
    # Create embedding function using a local model
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name
    )
    
    # Try to get the collection, create it if it doesn't exist
    try:
        return client.get_collection(
            name=collection_name,
            embedding_function=embedding_func # Assign the embedding function
        )
    except Exception:
        print(f"Collection '{collection_name}' not found. Creating new collection.")
        return client.create_collection(
            name=collection_name,
            embedding_function=embedding_func, # Assign the embedding function
            metadata={"hnsw:space": distance_function}
        )

# Removed: create_embeddings_batch and create_embedding functions

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Only attempt contextual embedding if MODEL_CHOICE and OPENAI_API_KEY are set
    if not model_choice or not openai.api_key:
        # print("MODEL_CHOICE or OPENAI_API_KEY not set. Skipping contextual embedding.")
        return chunk, False # Return original chunk and False flag
        
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

def add_documents_to_chroma(
    collection: chromadb.Collection, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 100 # Increased batch size for ChromaDB
) -> None:
    """
    Add documents to a ChromaDB collection in batches.
    
    Args:
        collection: ChromaDB collection
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of batches for adding documents
    """
    # Check if MODEL_CHOICE is set for contextual embeddings
    model_choice = os.getenv("MODEL_CHOICE")
    use_contextual_embeddings = bool(model_choice and openai.api_key) # Check both env vars

    ids = []
    documents_to_add = []
    metadatas_to_add = []
    
    # Prepare data for batch processing
    process_args = []
    for i in range(len(contents)):
        ids.append(f"{urls[i]}-{chunk_numbers[i]}") # Create unique ID for each chunk
        full_document = url_to_full_document.get(urls[i], "")
        process_args.append((urls[i], contents[i], full_document))
        metadatas_to_add.append(metadatas[i]) # Add original metadata for now

    # Apply contextual embedding to each chunk if enabled
    if use_contextual_embeddings:
        print("Generating contextual embeddings...")
        contextual_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks and collect results
            future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                            for idx, arg in enumerate(process_args)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result, success = future.result()
                    contextual_results.append((idx, result, success)) # Store index, result, success
                except Exception as e:
                    print(f"Error processing chunk {idx}: {e}")
                    # Use original content as fallback
                    contextual_results.append((idx, contents[idx], False)) # Store index, original content, False

        # Sort results back into original order and update documents/metadata
        contextual_results.sort(key=lambda x: x[0]) # Sort by original index
        
        for idx, contextual_text, success in contextual_results:
            documents_to_add.append(contextual_text)
            if success:
                metadatas_to_add[idx]["contextual_embedding"] = True
            else:
                 metadatas_to_add[idx]["contextual_embedding"] = False # Explicitly mark if failed

    else:
        # If not using contextual embeddings, use original contents
        documents_to_add = contents
        # Ensure metadata still includes the flag, even if false
        for meta in metadatas_to_add:
             meta["contextual_embedding"] = False


    if not documents_to_add:
        print("No documents to add to ChromaDB.")
        return

    print(f"Adding {len(documents_to_add)} chunks to ChromaDB collection '{collection.name}'...")

    # Add documents in batches
    for batch_start in range(0, len(documents_to_add), batch_size):
        batch_end = min(batch_start + batch_size, len(documents_to_add))
        
        collection.add(
            ids=ids[batch_start:batch_end],
            documents=documents_to_add[batch_start:batch_end],
            metadatas=metadatas_to_add[batch_start:batch_end],
            # Embeddings are generated automatically by the collection's embedding function
        )
    print(f"Successfully added {len(documents_to_add)} chunks to ChromaDB collection '{collection.name}'.")


def search_documents_chroma(
    collection: chromadb.Collection, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in ChromaDB using vector similarity.
    
    Args:
        collection: ChromaDB collection
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        
    Returns:
        List of matching documents
    """
    # Execute the search using the collection.query method
    try:
        # ChromaDB handles embedding the query text internally
        results = collection.query(
            query_texts=[query],
            n_results=match_count,
            where=filter_metadata, # Pass the dictionary directly
            include=["documents", "metadatas", "distances"]
        )
        
        # Format the results to match the expected output structure
        formatted_results = []
        if results and results.get("documents") and results["documents"][0]:
             for i in range(len(results["documents"][0])):
                 formatted_results.append({
                     "url": results["metadatas"][0][i].get("url"), # Assuming URL is in metadata
                     "content": results["documents"][0][i],
                     "metadata": results["metadatas"][0][i],
                     "similarity": 1 - results["distances"][0][i] # Convert distance to similarity (0 to 1)
                 })
        
        return formatted_results
    except Exception as e:
        print(f"Error searching documents in ChromaDB: {e}")
        return []
