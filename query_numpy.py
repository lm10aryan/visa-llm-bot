"""
RAG Pipeline - Step 2: Query using NumPy (NO FAISS)
This uses simple cosine similarity search
Works on M1 Macs without FAISS!
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    def __init__(self, rag_dir=None, config_path="config.yaml"):
        self.rag_dir = self._resolve_rag_dir(rag_dir, config_path)
        
        # Load embedding model
        print("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load embeddings
        embeddings_path = self.rag_dir / "embeddings.npy"
        metadata_path = self.rag_dir / "chunks_metadata.pkl"
        
        if not embeddings_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Embeddings or metadata not found in {self.rag_dir.resolve()}\n"
                "Please run the embedding pipeline or point RAGRetriever to the correct directory."
            )
        
        print(f"Loading embeddings from {embeddings_path}...")
        self.embeddings = np.load(embeddings_path)
        
        # Load chunks metadata
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"✅ RAG system loaded ({len(self.chunks)} chunks)")
    
    def _resolve_rag_dir(self, rag_dir, config_path):
        """Determine where embeddings/metadata live"""
        candidates = []
        if rag_dir:
            candidates.append(Path(rag_dir))
        
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            index_dir = config.get('rag', {}).get('index_dir', '')
            if index_dir:
                candidates.append(Path(index_dir))
        
        candidates.append(Path("."))
        
        for candidate in candidates:
            embeddings_path = candidate / "embeddings.npy"
            metadata_path = candidate / "chunks_metadata.pkl"
            if embeddings_path.exists() and metadata_path.exists():
                return candidate
        
        raise FileNotFoundError(
            "Unable to locate embeddings.npy and chunks_metadata.pkl. "
            "Specify rag_dir explicitly or update config.yaml -> rag.index_dir."
        )
    
    def cosine_similarity(self, query_embedding, embeddings):
        """Calculate cosine similarity between query and all embeddings"""
        # Already normalized, so dot product = cosine similarity
        similarities = np.dot(embeddings, query_embedding)
        return similarities
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k most relevant chunks for a query
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of dicts with 'text', 'source_url', 'source_title', 'score'
        """
        # Embed the query
        query_embedding = self.model.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # Calculate similarities
        similarities = self.cosine_similarity(query_embedding, self.embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Get the relevant chunks
        results = []
        for i, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            results.append({
                'rank': i + 1,
                'text': chunk['text'],
                'source_url': chunk['source_url'],
                'source_title': chunk['source_title'],
                'score': float(similarities[idx]),
                'chunk_id': chunk.get('chunk_id', f'chunk_{idx}')
            })
        
        return results
    
    def format_context(self, results):
        """Format retrieved chunks into context for LLM"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}] {result['source_title']}\n"
                f"URL: {result['source_url']}\n"
                f"{result['text']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def print_results(self, query, results):
        """Pretty print retrieval results"""
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)
        print()
        
        for result in results:
            print(f"[Rank {result['rank']}] Similarity: {result['score']:.4f}")
            print(f"Source: {result['source_title']}")
            print(f"URL: {result['source_url']}")
            print(f"Text: {result['text'][:200]}...")
            print("-" * 60)
            print()

def main():
    """Test the retrieval system"""
    if len(sys.argv) < 2:
        print("Usage: python query_numpy.py 'Your question here'")
        print("\nExample queries:")
        print("  python query_numpy.py 'What is F-1 OPT?'")
        print("  python query_numpy.py 'How does H-1B lottery work?'")
        print("  python query_numpy.py 'Can I work on a study permit in Canada?'")
        return
    
    query = sys.argv[1]
    
    # Initialize retriever
    retriever = RAGRetriever()
    
    # Retrieve results
    results = retriever.retrieve(query, top_k=5)
    
    # Print results
    retriever.print_results(query, results)
    
    # Also show formatted context (what will be sent to LLM)
    print("=" * 60)
    print("FORMATTED CONTEXT FOR LLM:")
    print("=" * 60)
    print()
    context = retriever.format_context(results)
    print(context)

if __name__ == "__main__":
    main()
