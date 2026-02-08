import faiss
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Optional, Tuple
import re
from rank_bm25 import BM25Okapi
import numpy as np
from pathlib import Path
import os

from .generator import Generator
from .config_loader import get_config, Config

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed

class RAG:
    """Retrieval-Augmented Generation system with hybrid search and reranking"""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: Optional[str] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        use_generator: bool = False
    ):
        """Initialize RAG system
        
        Args:
            config: Configuration object (if None, loads from default config file)
            model_name: Override embedding model name
            m: Override HNSW M parameter
            ef_construction: Override HNSW ef_construction parameter
            use_reranker: Override reranker usage
            use_generator: Whether to load generator
        """
        # Load configuration
        self.config = config or get_config()
        
        # Get model configs
        emb_config = self.config.models.get('embedding', {})
        rerank_config = self.config.models.get('reranker', {})
        
        # Use provided values or fall back to config
        self.model_name = model_name or emb_config.get('name', 'intfloat/e5-large-v2')
        self.is_e5_model = 'e5-' in self.model_name.lower()
        device = emb_config.get('device', 'cuda')
        
        # Get HuggingFace token if available
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        
        print(f"Loading embedding model: {self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name, device=device, local_files_only=True, token=hf_token)
        except Exception as e:
            print(f"  Model not cached locally, downloading from HuggingFace...")
            self.model = SentenceTransformer(self.model_name, device=device, token=hf_token)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Load reranker
        self.use_reranker = use_reranker if use_reranker is not None else rerank_config.get('enabled', True)
        if self.use_reranker:
            reranker_name = rerank_config.get('name', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
            reranker_device = rerank_config.get('device', 'cuda')
            print(f"Loading cross-encoder reranker: {reranker_name}...")
            try:
                self.reranker = CrossEncoder(reranker_name, device=reranker_device, local_files_only=True, token=hf_token)
            except Exception as e:
                print(f"  Model not cached locally, downloading from HuggingFace...")
                self.reranker = CrossEncoder(reranker_name, device=reranker_device, token=hf_token)
        else:
            self.reranker = None
        
        # Load generator
        self.use_generator = use_generator
        if use_generator:
            print("Initializing generator...")
            self.generator = Generator(config=self.config)
        else:
            self.generator = None
        
        # HNSW parameters
        self.m = m or self.config.indexing.hnsw.m
        self.ef_construction = ef_construction or self.config.indexing.hnsw.ef_construction
        
        # Storage for documents and index
        self.documents = []
        self.chunks = []  # Store individual chunks with metadata
        self.index = None
        self.bm25 = None
        self.section_embeddings = None  # Embeddings for sections/headers
        self.section_list = []  # List of unique sections
        self.indexed_pdfs = {}  # Track PDFs: {pdf_name: {title, pages, chunks}}
        
    def load_data(self, json_paths, titles_override: Optional[Dict[str, str]] = None):
        """Load data from one or multiple JSON files
        
        Args:
            json_paths: Path or list of paths to JSON files
            titles_override: Optional dict mapping json_path -> title to override titles from files
        """
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        
        titles_override = titles_override or {}
        
        print(f"Loading data from {len(json_paths)} file(s)...")
        for json_path in json_paths:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            pdf_name = data.get('pdf_name', 'unknown')
            pages = data.get('pages', [])
            
            # Use override title if provided, otherwise extract from metadata
            if json_path in titles_override:
                title = titles_override[json_path]
            elif str(json_path) in titles_override:
                title = titles_override[str(json_path)]
            else:
                title = data.get('title', 'Unknown')
            
            # Track this PDF
            doc_start_idx = len(self.documents)
            
            for page in pages:
                self.documents.append({
                    'pdf_name': pdf_name,
                    'title': title,
                    'page': page['page'],
                    'content': page['text'],
                    'word_count': len(page['text'].split()),
                    'hierarchy_path': page.get('hierarchy_path', []),
                    'hierarchy_context': page.get('hierarchy_context', ''),
                    'metadata': page.get('metadata', {}),
                    'headers': page.get('metadata', {}).get('headers', []),
                    'section': page.get('metadata', {}).get('section', None)
                })
            
            self.indexed_pdfs[pdf_name] = {
                'title': title,
                'pages': len(pages),
                'doc_indices': (doc_start_idx, len(self.documents)),
                'path': json_path
            }
            
            print(f"  ✓ {pdf_name}: {len(pages)} pages (Title: {title})")
        
        print(f"\nTotal loaded: {len(self.documents)} pages from {len(self.indexed_pdfs)} PDFs")
        
        # Print summary of detected headers
        total_headers = sum(len(doc.get('headers', [])) for doc in self.documents)
        sections = set(doc.get('section') for doc in self.documents if doc.get('section'))
        print(f"Detected {total_headers} headers across {len(sections)} sections")    
    
    def _encode_texts(self, texts: List[str], show_progress: bool = False, is_query: bool = False) -> np.ndarray:
        """Encode texts using SentenceTransformer.
        
        Args:
            texts: List of texts to encode
            show_progress: Show progress bar
            is_query: Whether texts are queries (for E5 models, adds 'query: ' prefix)
        """
        # Add E5 model prefixes if needed
        if self.is_e5_model:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + text for text in texts]
        return self.model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        headers: Optional[List[Dict]] = None
    ) -> List[str]:
        """Chunk text while respecting paragraph and section boundaries.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size in words
            overlap: Overlap size in words
            headers: List of headers from the page to avoid splitting
        
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunk_config = self.config.document_processing.chunking
        
        # First split by double newlines (paragraphs) or single newlines
        paragraphs = re.split(r'\n\n+', text)
        if len(paragraphs) == 1:
            paragraphs = re.split(r'\n+', text)
        
        # Further split paragraphs into sentences
        segments = []
        header_texts = {h.get('text', '').strip() for h in (headers or [])} if headers else set()
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this paragraph is a header - keep it intact
            is_header = para in header_texts or (
                len(para.split()) <= chunk_config.max_header_words and 
                para[0].isupper() and 
                not para.endswith(('.', '!', '?'))
            )
            
            if is_header:
                segments.append({'text': para, 'type': 'header'})
            else:
                # Split paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if sent.strip():
                        segments.append({'text': sent.strip(), 'type': 'sentence'})
        
        # Build chunks respecting boundaries
        chunks = []
        current_chunk = []
        current_length = 0
        last_was_header = False
        
        for i, segment in enumerate(segments):
            segment_text = segment['text']
            segment_length = len(segment_text.split())
            is_header = segment['type'] == 'header'
            
            # Check if we should start a new chunk
            should_split = False
            min_size = chunk_size * chunk_config.min_chunk_ratio
            max_size = chunk_size * chunk_config.max_chunk_ratio
            
            if current_length + segment_length > chunk_size and current_chunk:
                # Don't split right before a header - save it for next chunk
                if is_header:
                    should_split = True
                # Don't split if current chunk is too small
                elif current_length >= min_size:
                    should_split = True
                # If adding this would make chunk too large, split
                elif current_length + segment_length > max_size:
                    should_split = True
            
            if should_split:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap (only if not starting with header)
                if overlap > 0 and not is_header and len(current_chunk) > 0:
                    # Calculate how many sentences to keep for overlap
                    overlap_sents = []
                    overlap_words = 0
                    for sent in reversed(current_chunk):
                        sent_words = len(sent.split())
                        if overlap_words + sent_words <= overlap:
                            overlap_sents.insert(0, sent)
                            overlap_words += sent_words
                        else:
                            break
                    current_chunk = overlap_sents
                    current_length = overlap_words
                else:
                    current_chunk = []
                    current_length = 0
            
            # Add segment to current chunk
            current_chunk.append(segment_text)
            current_length += segment_length
            last_was_header = is_header
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Fallback for short texts
        return chunks if chunks else [text[:chunk_config.fallback_max_chars]]
    
    def _build_chunks(
        self,
        use_summary: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> None:
        """Build chunks from loaded documents (internal helper)
        
        Args:
            use_summary: Use document summaries instead of full text
            chunk_size: Chunk size in words (uses config default if None)
            overlap: Overlap size in words (uses config default if None)
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_data() first.")
        
        # Use config defaults if not provided
        use_summary = use_summary if use_summary is not None else self.config.indexing.use_summary
        chunk_size = chunk_size or self.config.indexing.chunk_size
        overlap = overlap or self.config.indexing.overlap
        
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
                chunks = self._chunk_text(doc['content'], chunk_size, overlap, headers)
            else:
                chunks = [f"Page {page_num}"]
            
            for chunk_idx, chunk_text in enumerate(chunks):
                self.chunks.append({
                    'text': chunk_text,  # Use original text for embedding (no hierarchy prepending)
                    'hierarchy_context': hierarchy_context,
                    'doc_idx': doc_idx,
                    'chunk_idx': chunk_idx,
                    'pdf_name': pdf_name,
                    'title': doc.get('title', 'Unknown'),
                    'page': page_num,
                    'total_chunks': len(chunks),
                    'section': section,
                    'headers': headers,
                    'metadata': metadata
                })
        
        print(f"Created {len(self.chunks)} chunks from {len(self.documents)} documents")
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from existing chunks"""
        if not self.chunks:
            raise ValueError("No chunks available. Call _build_chunks() first.")
        
        print("Building BM25 index...")
        tokenized_corpus = [chunk['text'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built")
    
    def _build_section_embeddings(self) -> None:
        """Build section embeddings from existing chunks"""
        if not self.chunks:
            raise ValueError("No chunks available. Call _build_chunks() first.")
        
        print("Building section embeddings...")
        sections_set = set()
        for chunk in self.chunks:
            if chunk.get('section'):
                sections_set.add(chunk['section'])
            # Also add header texts
            for header in chunk.get('headers', []):
                if header.get('text'):
                    sections_set.add(header['text'])
        
        self.section_list = list(sections_set)
        if self.section_list:
            self.section_embeddings = self._encode_texts(self.section_list, show_progress=False)
            faiss.normalize_L2(self.section_embeddings)
            print(f"Embedded {len(self.section_list)} unique sections/headers")
        else:
            self.section_embeddings = None
            print("No sections found to embed")
        
    def build_index(
        self,
        use_summary: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> None:
        """Build FAISS index and BM25 index from loaded documents
        
        Args:
            use_summary: Use document summaries instead of full text
            chunk_size: Chunk size in words (uses config default if None)
            overlap: Overlap size in words (uses config default if None)
        """
        # Build chunks
        self._build_chunks(use_summary, chunk_size, overlap)
        
        # Generate embeddings for all chunks
        print("Generating embeddings...")
        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self._encode_texts(texts, show_progress=True)
        
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
        self._build_bm25_index()
        
        # Build section embeddings
        self._build_section_embeddings()
    
    def _compute_section_boost(self, query_emb: np.ndarray, chunk: Dict) -> float:
        """Compute semantic similarity boost based on section/header relevance.
        
        Args:
            query_emb: Pre-computed normalized query embedding
            chunk: Chunk dictionary with section/headers
        
        Returns:
            Section boost score
        """
        if self.section_embeddings is None or not self.section_list:
            return 0.0
        
        # Find best matching section for this chunk
        max_similarity = 0.0
        
        # Check chunk's section
        if chunk.get('section') and chunk['section'] in self.section_list:
            section_idx = self.section_list.index(chunk['section'])
            section_emb = self.section_embeddings[section_idx:section_idx+1]
            similarity = float((query_emb @ section_emb.T)[0][0])
            max_similarity = max(max_similarity, similarity)
        
        # Check chunk's headers
        for header in chunk.get('headers', []):
            header_text = header.get('text')
            if header_text and header_text in self.section_list:
                header_idx = self.section_list.index(header_text)
                header_emb = self.section_embeddings[header_idx:header_idx+1]
                similarity = float((query_emb @ header_emb.T)[0][0])
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 3) -> List[Dict]:
        """Rerank results using cross-encoder and add section semantic boost
        
        Args:
            query: Search query
            results: List of search results
            top_k: Number of top results to return
        
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        section_boost_weight = self.config.search.section_boost_weight
        
        # Compute query embedding once for section boost
        query_emb = None
        if self.section_embeddings is not None:
            query_emb = self._encode_texts([query], show_progress=False, is_query=True)
            faiss.normalize_L2(query_emb)
        
        if self.reranker:
            # Prepare query-document pairs for reranker
            pairs = [[query, r['chunk']] for r in results]
            
            # Get reranker scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Add rerank scores and section boost
            for i, result in enumerate(results):
                result['rerank_score'] = float(rerank_scores[i])
                # Add section semantic boost
                if query_emb is not None:
                    section_boost = self._compute_section_boost(query_emb, result)
                    result['section_boost'] = section_boost
                    result['final_score'] = result['rerank_score'] + (section_boost * section_boost_weight)
                else:
                    result['section_boost'] = 0.0
                    result['final_score'] = result['rerank_score']
            
            reranked = sorted(results, key=lambda x: x['final_score'], reverse=True)
        else:
            # No reranker, just use section boost if available
            for result in results:
                if query_emb is not None:
                    section_boost = self._compute_section_boost(query_emb, result)
                    result['section_boost'] = section_boost
                    result['final_score'] = section_boost
                else:
                    result['section_boost'] = 0.0
                    result['final_score'] = 0.0
            
            reranked = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        return reranked[:top_k]
        
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        ef_search: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        title_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search for relevant chunks using cosine similarity with optional title filter
        
        Args:
            query: Search query
            top_k: Number of results to return (uses config default if None)
            ef_search: HNSW ef_search parameter (uses config default if None)
            use_reranker: Whether to use reranker (uses instance setting if None)
            title_filter: Filter results by document title
        
        Returns:
            List of search results
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Use config defaults
        top_k = top_k or self.config.search.top_k
        ef_search = ef_search or self.config.indexing.hnsw.ef_search
        use_reranker = use_reranker if use_reranker is not None else self.use_reranker
        
        # Get more candidates if using reranker or filtering
        retrieve_multiplier = self.config.search.retrieve_multiplier
        retrieve_k = top_k * retrieve_multiplier if use_reranker and self.reranker else top_k
        if title_filter:
            filter_mult = self.config.search.title_filter_multiplier
            retrieve_k = min(retrieve_k * filter_mult, len(self.chunks))
        
        self.index.hnsw.efSearch = ef_search
        
        query_embedding = self._encode_texts([query], show_progress=False, is_query=True)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, retrieve_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                
                # Apply title filter (fuzzy match - check if filter is substring of title)
                if title_filter and title_filter.lower() not in chunk.get('title', '').lower():
                    continue
                
                doc = self.documents[chunk['doc_idx']]
                cosine_similarity = 1 - (dist ** 2) / 2
                display_text = chunk['text']
                
                result = {
                    'rank': len(results) + 1,
                    'cosine_similarity': float(cosine_similarity),
                    'l2_distance': float(dist),
                    'pdf_name': chunk['pdf_name'],
                    'title': chunk.get('title', 'Unknown'),
                    'page': chunk['page'],
                    'chunk': display_text[:500] + '...' if len(display_text) > 500 else display_text,
                    'chunk_info': f"{chunk['chunk_idx'] + 1}/{chunk['total_chunks']}",
                    'word_count': doc.get('word_count', 0),
                    'section': chunk.get('section', None),
                    'headers': chunk.get('headers', []),
                    'hierarchy': chunk.get('hierarchy_context', '')
                }
                results.append(result)
                
                # Stop if we have enough results
                if len(results) >= retrieve_k:
                    break
        
        if use_reranker and self.reranker:
            results = self.rerank(query, results, top_k)
        else:
            results = results[:top_k]
        
        return results
    
    def search_bm25(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        title_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search using BM25 algorithm with optional title filter.
        
        Args:
            query: Search query
            top_k: Number of results to return (uses config default if None)
            use_reranker: Whether to use reranker (uses instance setting if None)
            title_filter: Filter results by document title
        
        Returns:
            List of search results
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        # Use config defaults
        top_k = top_k or self.config.search.top_k
        use_reranker = use_reranker if use_reranker is not None else self.use_reranker
        
        # Get more candidates if using reranker or filtering
        retrieve_multiplier = self.config.search.retrieve_multiplier
        retrieve_k = top_k * retrieve_multiplier if use_reranker and self.reranker else top_k
        if title_filter:
            filter_mult = self.config.search.title_filter_multiplier
            retrieve_k = min(retrieve_k * filter_mult, len(self.chunks))
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                
                # Apply title filter (fuzzy match - check if filter is substring of title)
                if title_filter and title_filter.lower() not in chunk.get('title', '').lower():
                    continue
                
                doc = self.documents[chunk['doc_idx']]
                display_text = chunk['text']
                
                result = {
                    'rank': len(results) + 1,
                    'bm25_score': float(scores[idx]),
                    'pdf_name': chunk['pdf_name'],
                    'title': chunk.get('title', 'Unknown'),
                    'page': chunk['page'],
                    'chunk': display_text[:500] + '...' if len(display_text) > 500 else display_text,
                    'chunk_info': f"{chunk['chunk_idx'] + 1}/{chunk['total_chunks']}",
                    'word_count': doc.get('word_count', 0),
                    'section': chunk.get('section', None),
                    'headers': chunk.get('headers', []),
                    'hierarchy': chunk.get('hierarchy_context', '')
                }
                results.append(result)
                
                # Stop if we have enough results
                if len(results) >= retrieve_k:
                    break
        
        # Rerank if enabled
        if use_reranker and self.reranker:
            return self.rerank(query, results, top_k)
        else:
            return results[:top_k]
    
    def search_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidates_per_method: Optional[int] = None,
        emb_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        use_reranker: Optional[bool] = None,
        title_filter: Optional[str] = None
    ) -> List[Dict]:
        """Hybrid search combining BM25 and embedding with optional title filter
        
        Args:
            query: Search query
            top_k: Number of final results (uses config default if None)
            candidates_per_method: Number of candidates from each method (uses config default if None)
            emb_weight: Weight for embedding scores (uses config default if None)
            bm25_weight: Weight for BM25 scores (uses config default if None)
            use_reranker: Whether to use reranker (uses instance setting if None)
            title_filter: Filter results by document title
        
        Returns:
            List of hybrid search results
        """
        # Use config defaults
        top_k = top_k or self.config.search.top_k
        use_reranker = use_reranker if use_reranker is not None else self.use_reranker
        hybrid_config = self.config.search.hybrid
        candidates_per_method = candidates_per_method or hybrid_config.candidates_per_method
        emb_weight = emb_weight if emb_weight is not None else hybrid_config.embedding_weight
        bm25_weight = bm25_weight if bm25_weight is not None else hybrid_config.bm25_weight
        # Get candidates from both methods (without reranking yet)
        emb_results = self.search(query, top_k=candidates_per_method, ef_search=100, use_reranker=False, title_filter=title_filter)
        bm25_results = self.search_bm25(query, top_k=candidates_per_method, use_reranker=False, title_filter=title_filter)
        
        # Normalize scores using min-max normalization
        def normalize_scores(results, score_key):
            if not results:
                return results
            scores = [r[score_key] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            for r in results:
                if score_range > 0:
                    r[f'{score_key}_normalized'] = (r[score_key] - min_score) / score_range
                else:
                    r[f'{score_key}_normalized'] = 1.0
            return results
        
        emb_results = normalize_scores(emb_results, 'cosine_similarity')
        bm25_results = normalize_scores(bm25_results, 'bm25_score')
        
        # Combine and remove duplicates based on page + chunk position
        seen = {}
        combined = []
        
        # Process embedding results first
        for r in emb_results:
            key = (r['page'], r['chunk'][:100])
            if key not in seen:
                seen[key] = r
                r['hybrid_score'] = r['cosine_similarity_normalized'] * emb_weight
                r['chunk_id'] = key
                combined.append(r)
        
        # Add or update with BM25 results
        for r in bm25_results:
            key = (r['page'], r['chunk'][:100])
            if key in seen:
                # Update existing result with BM25 score
                existing = seen[key]
                existing['bm25_score'] = r['bm25_score']
                existing['bm25_score_normalized'] = r['bm25_score_normalized']
                existing['hybrid_score'] += r['bm25_score_normalized'] * bm25_weight
            else:
                # New result from BM25 only
                seen[key] = r
                r['hybrid_score'] = r['bm25_score_normalized'] * bm25_weight
                r['chunk_id'] = key
                combined.append(r)
        
        # Sort by hybrid score
        combined = sorted(combined, key=lambda x: x['hybrid_score'], reverse=True)
        
        # Rerank the combined results if enabled
        if use_reranker and self.reranker:
            return self.rerank(query, combined, top_k)
        else:
            # Update ranks
            for i, r in enumerate(combined[:top_k]):
                r['rank'] = i + 1
            return combined[:top_k]
    
    def save_index(self, path: str = 'faiss_hnsw.index'):
        """Save the FAISS index to disk."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        faiss.write_index(self.index, path)
        print(f"Index saved to {path}")
        
    def load_index(
        self,
        path: str = 'faiss_hnsw.index',
        rebuild_chunks: bool = True,
        rebuild_bm25: bool = True,
        rebuild_section_embeddings: bool = False
    ):
        """Load a FAISS index from disk and optionally rebuild chunks/BM25
        
        Args:
            path: Path to FAISS index file
            rebuild_chunks: Whether to rebuild chunks from documents (required for BM25)
            rebuild_bm25: Whether to rebuild BM25 index (requires chunks)
            rebuild_section_embeddings: Whether to rebuild section embeddings (slow)
        """
        self.index = faiss.read_index(path)
        print(f"Index loaded from {path} with {self.index.ntotal} vectors")
        
        # Rebuild chunks if requested and we have documents
        if rebuild_chunks and self.documents and not self.chunks:
            self._build_chunks()
        
        # Rebuild BM25 if requested and we have chunks
        if rebuild_bm25 and self.chunks:
            self._build_bm25_index()
        
        # Build section embeddings only if explicitly requested
        if rebuild_section_embeddings and self.chunks and not self.section_embeddings:
            self._build_section_embeddings()
    
    def answer(
        self,
        query: str,
        search_method: str = 'hybrid',
        top_k: Optional[int] = None,
        title_filter: Optional[str] = None,
        auto_detect_title: bool = False
    ) -> str:
        """Generate answer using retrieval and LLM with conversation context
        
        Args:
            query: User query
            search_method: 'hybrid', 'bm25', or 'semantic'
            top_k: Number of chunks to retrieve (uses config default if None)
            title_filter: Explicit document title filter (overrides auto-detection)
            auto_detect_title: Whether to auto-detect title from query
        
        Returns:
            Generated answer
        """
        if not self.use_generator or self.generator is None:
            raise ValueError("Generator not initialized. Set use_generator=True")
        
        # Use config default for top_k
        top_k = top_k or self.config.generation.max_context_chunks
        
        # Auto-detect title if enabled and no explicit filter
        if auto_detect_title and not title_filter:
            detected_title = self.detect_title_in_query(query)
            if detected_title:
                title_filter = detected_title
                print(f"[Auto-detected title filter: {detected_title}]")
        
        if search_method == 'hybrid':
            chunks = self.search_hybrid(query, top_k=top_k, use_reranker=True, title_filter=title_filter)
        elif search_method == 'bm25':
            chunks = self.search_bm25(query, top_k=top_k, use_reranker=True, title_filter=title_filter)
        else:
            chunks = self.search(query, top_k=top_k, use_reranker=True, title_filter=title_filter)
        
        if not chunks:
            return f"No relevant information found{f' for {title_filter}' if title_filter else ''}."
        
        filter_msg = f" from {title_filter}" if title_filter else ""
        print(f"\n[Retrieved {len(chunks)} chunks{filter_msg} via {search_method}]")
        result = self.generator.generate(query, chunks)
        
        print("\nSources:")
        for i, src in enumerate(result['sources'][:5], 1):
            print(f"  {i}. {src['pdf']} - Page {src['page']} ({src.get('title', 'N/A')})")
        
        return result['answer']
    
    def clear_conversation(self):
        """Clear conversation history"""
        if self.generator:
            self.generator.clear_history()
    
    def get_indexed_pdfs(self) -> Dict[str, Dict]:
        """Get information about indexed PDFs"""
        return self.indexed_pdfs
    
    def get_titles(self) -> List[str]:
        """Get list of all unique document titles in the indexed data"""
        titles = set()
        for info in self.indexed_pdfs.values():
            titles.add(info['title'])
        return sorted(list(titles))
    
    def detect_title_in_query(self, query: str) -> Optional[str]:
        """Detect document title mentioned in query
        
        Args:
            query: User query text
            
        Returns:
            Document title if detected, None otherwise
        """
        titles = self.get_titles()
        query_lower = query.lower()
        
        # Check for exact matches (case-insensitive)
        for title in titles:
            if title.lower() in query_lower:
                return title
        
        # Check for partial matches (e.g., keywords from title)
        for title in titles:
            title_words = title.lower().split()
            for word in title_words:
                if len(word) > 3 and word in query_lower:  # Match words longer than 3 chars
                    return title
        
        return None
    
    def search_with_auto_filter(self, query: str, top_k: int = 5, 
                                search_method: str = 'hybrid',
                                use_reranker: bool = True) -> List[Dict]:
        """Search with automatic document title detection from query
        
        Args:
            query: User query
            top_k: Number of results to return
            search_method: 'hybrid', 'bm25', or 'semantic'
            use_reranker: Whether to use reranker
            
        Returns:
            List of search results with detected title info
        """
        # Try to detect title from query
        detected_title = self.detect_title_in_query(query)
        
        if detected_title:
            print(f"[Auto-detected title filter: {detected_title}]")
        
        # Perform search with detected filter
        if search_method == 'hybrid':
            results = self.search_hybrid(query, top_k=top_k, use_reranker=use_reranker, 
                                        title_filter=detected_title)
        elif search_method == 'bm25':
            results = self.search_bm25(query, top_k=top_k, use_reranker=use_reranker,
                                      title_filter=detected_title)
        else:
            results = self.search(query, top_k=top_k, use_reranker=use_reranker,
                                 title_filter=detected_title)
        
        # Add detection info to results
        for result in results:
            result['auto_filtered'] = detected_title is not None
            result['detected_title'] = detected_title
        
        return results