import faiss
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import re
from rank_bm25 import BM25Okapi

class RAG:    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', m: int = 32, ef_construction: int = 200):        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device="cuda")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # HNSW parameters
        self.m = m
        self.ef_construction = ef_construction
        
        # Storage for documents and index
        self.documents = []
        self.chunks = []  # Store individual chunks with metadata
        self.index = None
        self.bm25 = None
        
    def load_data(self, json_path: str):
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pdf_name = data.get('pdf_name', 'unknown')
        pages = data.get('pages', [])
        
        self.documents = [{
            'pdf_name': pdf_name,
            'page': page['page'],
            'content': page['text'],
            'word_count': len(page['text'].split()),
            'hierarchy_path': page.get('hierarchy_path', []),
            'hierarchy_context': page.get('hierarchy_context', ''),
            'metadata': page.get('metadata', {}),
            'headers': page.get('metadata', {}).get('headers', []),
            'section': page.get('metadata', {}).get('section', None)
        } for page in pages]
        
        print(f"Loaded {len(self.documents)} pages from {pdf_name}")
        
        # Print summary of detected headers
        total_headers = sum(len(doc.get('headers', [])) for doc in self.documents)
        sections = set(doc.get('section') for doc in self.documents if doc.get('section'))
        print(f"Detected {total_headers} headers across {len(sections)} sections")
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        if not text:
            return []
        
        # Split by sentences (basic approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > 1:
                    # Keep last few sentences for overlap
                    overlap_text = ' '.join(current_chunk[-2:])
                    overlap_words = len(overlap_text.split())
                    if overlap_words <= overlap:
                        current_chunk = current_chunk[-2:]
                        current_length = overlap_words
                    else:
                        current_chunk = []
                        current_length = 0
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text[:1000]]  # Fallback for short texts
        
    def build_index(self, use_summary: bool = False, chunk_size: int = 512, overlap: int = 50):
        if not self.documents:
            raise ValueError("No documents loaded. Call load_data() first.")
        
        print("Chunking documents...")
        self.chunks = []
        
        for doc_idx, doc in enumerate(self.documents):
            pdf_name = doc.get('pdf_name', 'N/A')
            page_num = doc.get('page', 0)
            section = doc.get('section', None)
            headers = doc.get('headers', [])
            metadata = doc.get('metadata', {})
            hierarchy_context = doc.get('hierarchy_context', '')
            
            if use_summary and 'summary' in doc and doc['summary']:
                chunks = [doc['summary']]
            elif 'content' in doc and doc['content']:
                chunks = self._chunk_text(doc['content'], chunk_size, overlap)
            else:
                chunks = [f"Page {page_num}"]
            
            for chunk_idx, chunk_text in enumerate(chunks):
                # Prepend hierarchy context to chunk for embedding
                chunk_with_context = hierarchy_context + chunk_text
                
                self.chunks.append({
                    'text': chunk_with_context,  # Text WITH context for embedding
                    'original_text': chunk_text,  # Original text without context for display
                    'hierarchy_context': hierarchy_context,
                    'doc_idx': doc_idx,
                    'chunk_idx': chunk_idx,
                    'pdf_name': pdf_name,
                    'page': page_num,
                    'total_chunks': len(chunks),
                    'section': section,
                    'headers': headers,
                    'metadata': metadata
                })
        
        print(f"Created {len(self.chunks)} chunks from {len(self.documents)} documents")
        
        # Generate embeddings for all chunks
        print("Generating embeddings...")
        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        print("Normalizing embeddings for cosine similarity...")
        faiss.normalize_L2(embeddings)
        
        print("Building FAISS HNSW index...")
        # Create HNSW index (with normalized vectors, L2 distance = cosine similarity)
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.m)
        self.index.hnsw.efConstruction = self.ef_construction
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        print(f"Index built with {self.index.ntotal} vectors")
        
        # Build BM25 index
        print("Building BM25 index...")
        tokenized_corpus = [chunk['text'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built")
        
    def search(self, query: str, top_k: int = 5, ef_search: int = 50) -> List[Dict]:
        """
        Search for relevant chunks given a query using cosine similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            ef_search: Size of the dynamic candidate list during search
            
        Returns:
            List of dictionaries containing search results with cosine similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Set search parameter
        self.index.hnsw.efSearch = ef_search
        
        # Generate and normalize query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search (with normalized vectors, L2 distance corresponds to cosine similarity)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                doc = self.documents[chunk['doc_idx']]
                
                cosine_similarity = 1 - (dist ** 2) / 2
                
                # Use original text for display (without hierarchy context)
                display_text = chunk.get('original_text', chunk['text'])
                
                result = {
                    'rank': i + 1,
                    'cosine_similarity': float(cosine_similarity),
                    'l2_distance': float(dist),
                    'pdf_name': chunk['pdf_name'],
                    'page': chunk['page'],
                    'chunk': display_text[:500] + '...' if len(display_text) > 500 else display_text,
                    'chunk_info': f"{chunk['chunk_idx'] + 1}/{chunk['total_chunks']}",
                    'word_count': doc.get('word_count', 0),
                    'section': chunk.get('section', None),
                    'headers': chunk.get('headers', []),
                    'hierarchy': chunk.get('hierarchy_context', '')
                }
                results.append(result)
        
        return results
    
    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using BM25 algorithm."""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                doc = self.documents[chunk['doc_idx']]
                
                # Use original text for display
                display_text = chunk.get('original_text', chunk['text'])
                
                result = {
                    'rank': i + 1,
                    'bm25_score': float(scores[idx]),
                    'pdf_name': chunk['pdf_name'],
                    'page': chunk['page'],
                    'chunk': display_text[:500] + '...' if len(display_text) > 500 else display_text,
                    'chunk_info': f"{chunk['chunk_idx'] + 1}/{chunk['total_chunks']}",
                    'word_count': doc.get('word_count', 0),
                    'section': chunk.get('section', None),
                    'headers': chunk.get('headers', []),
                    'hierarchy': chunk.get('hierarchy_context', '')
                }
                results.append(result)
        
        return results
    
    def save_index(self, path: str = 'faiss_hnsw.index'):
        """Save the FAISS index to disk."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        faiss.write_index(self.index, path)
        print(f"Index saved to {path}")
        
    def load_index(self, path: str = 'faiss_hnsw.index'):
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(path)
        print(f"Index loaded from {path} with {self.index.ntotal} vectors")