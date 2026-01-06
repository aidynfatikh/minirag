import faiss
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Optional
import re
from rank_bm25 import BM25Okapi
import numpy as np
from generator import Generator

class RAG:    
    def __init__(self, model_name: str = 'intfloat/e5-large-v2', m: int = 32, ef_construction: int = 200, use_reranker: bool = True, use_generator: bool = False):        
        self.model_name = model_name
        self.is_e5_model = 'e5-' in model_name.lower()
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device="cuda")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Load reranker
        self.use_reranker = use_reranker
        if use_reranker:
            print("Loading cross-encoder reranker: ms-marco-MiniLM-L-12-v2...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device="cuda")
        else:
            self.reranker = None
        
        # Load generator
        self.use_generator = use_generator
        if use_generator:
            print("Initializing generator...")
            self.generator = Generator()
        else:
            self.generator = None
        
        # HNSW parameters
        self.m = m
        self.ef_construction = ef_construction
        
        # Storage for documents and index
        self.documents = []
        self.chunks = []  # Store individual chunks with metadata
        self.index = None
        self.bm25 = None
        self.section_embeddings = None  # Embeddings for sections/headers
        self.section_list = []  # List of unique sections
        self.indexed_pdfs = {}  # Track PDFs: {pdf_name: {company, pages, chunks}}
        
    def load_data(self, json_paths):
        """Load data from one or multiple JSON files"""
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        
        print(f"Loading data from {len(json_paths)} file(s)...")
        for json_path in json_paths:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            pdf_name = data.get('pdf_name', 'unknown')
            pages = data.get('pages', [])
            
            # Extract company from metadata with improved detection
            company = data.get('company', 'Unknown')
            if not company or company == 'Unknown':
                # Try to extract from first page content
                if pages and 'text' in pages[0]:
                    first_text = pages[0]['text']
                    
                    # Look for common company patterns in first 15 lines
                    lines = first_text.split('\n')[:15]
                    for line in lines:
                        line = line.strip()
                        line_upper = line.upper()
                        
                        # Check for company indicators
                        if any(word in line_upper for word in ['CORPORATION', 'COMPANY', 'INC.', 'INC', 'LTD', 'GROUP', 'CO.']):
                            # Common company name patterns
                            if 'SEIKO EPSON CORPORATION' in line_upper:
                                company = 'Epson'
                                break
                            elif 'ANNUAL REPORT' not in line_upper and len(line) < 100:
                                # Clean the company name
                                company = line.replace('CORPORATION', '').replace('COMPANY', '')
                                company = company.replace('INC.', '').replace('INC', '')
                                company = company.replace('LTD.', '').replace('LTD', '')
                                company = company.strip()
                                if len(company) > 2 and len(company) < 60:
                                    break
                    
                    # Try metadata section/headers if still Unknown
                    if company == 'Unknown' and pages[0].get('metadata', {}).get('headers'):
                        headers = pages[0]['metadata']['headers']
                        for header in headers[:3]:  # Check first 3 headers
                            text = header.get('text', '').strip()
                            if any(word in text.upper() for word in ['CORPORATION', 'COMPANY', 'INC', 'GROUP']):
                                company = text
                                break
            
            # Track this PDF
            doc_start_idx = len(self.documents)
            
            for page in pages:
                self.documents.append({
                    'pdf_name': pdf_name,
                    'company': company,
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
                'company': company,
                'pages': len(pages),
                'doc_indices': (doc_start_idx, len(self.documents)),
                'path': json_path
            }
            
            print(f"  ✓ {pdf_name}: {len(pages)} pages (Company: {company})")
        
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
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50, headers: List[Dict] = None) -> List[str]:
        """Chunk text while respecting paragraph and section boundaries.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size in words
            overlap: Overlap size in words
            headers: List of headers from the page to avoid splitting
        """
        if not text:
            return []
        
        # First split by double newlines (paragraphs) or single newlines
        paragraphs = re.split(r'\n\n+', text)
        if len(paragraphs) == 1:
            # If no double newlines, try single newlines
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
                len(para.split()) <= 15 and 
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
            
            if current_length + segment_length > chunk_size and current_chunk:
                # Don't split right before a header - save it for next chunk
                if is_header:
                    should_split = True
                # Don't split if current chunk is too small
                elif current_length >= chunk_size * 0.5:
                    should_split = True
                # If adding this would make chunk too large, split
                elif current_length + segment_length > chunk_size * 1.5:
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
                    'company': doc.get('company', 'Unknown'),
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
        print("Building BM25 index...")
        tokenized_corpus = [chunk['text'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built")
        
        # Build section embeddings for semantic section matching
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
    
    def _compute_section_boost(self, query_emb: np.ndarray, chunk: Dict) -> float:
        """Compute semantic similarity boost based on section/header relevance.
        
        Args:
            query_emb: Pre-computed normalized query embedding
            chunk: Chunk dictionary with section/headers
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
        """Rerank results using cross-encoder and add section semantic boost"""
        if not results:
            return []
        
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
                    result['final_score'] = result['rerank_score'] + (section_boost * 2.0)
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
        
    def search(self, query: str, top_k: int = 5, ef_search: int = 50, use_reranker: bool = True, company_filter: Optional[str] = None) -> List[Dict]:
        """Search for relevant chunks using cosine similarity with optional company filter"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Get more candidates if using reranker or filtering
        retrieve_k = top_k * 4 if use_reranker and self.reranker else top_k
        if company_filter:
            retrieve_k = min(retrieve_k * 3, len(self.chunks))  # Get more candidates for filtering
        
        self.index.hnsw.efSearch = ef_search
        
        query_embedding = self._encode_texts([query], show_progress=False, is_query=True)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, retrieve_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                
                # Apply company filter
                if company_filter and chunk.get('company') != company_filter:
                    continue
                
                doc = self.documents[chunk['doc_idx']]
                cosine_similarity = 1 - (dist ** 2) / 2
                display_text = chunk['text']
                
                result = {
                    'rank': len(results) + 1,
                    'cosine_similarity': float(cosine_similarity),
                    'l2_distance': float(dist),
                    'pdf_name': chunk['pdf_name'],
                    'company': chunk.get('company', 'Unknown'),
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
    
    def search_bm25(self, query: str, top_k: int = 5, use_reranker: bool = True, company_filter: Optional[str] = None) -> List[Dict]:
        """Search using BM25 algorithm with optional company filter."""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        # Get more candidates if using reranker or filtering
        retrieve_k = top_k * 4 if use_reranker and self.reranker else top_k
        if company_filter:
            retrieve_k = min(retrieve_k * 3, len(self.chunks))  # Get more candidates for filtering
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                
                # Apply company filter
                if company_filter and chunk.get('company') != company_filter:
                    continue
                
                doc = self.documents[chunk['doc_idx']]
                display_text = chunk['text']
                
                result = {
                    'rank': len(results) + 1,
                    'bm25_score': float(scores[idx]),
                    'pdf_name': chunk['pdf_name'],
                    'company': chunk.get('company', 'Unknown'),
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
    
    def search_hybrid(self, query: str, top_k: int = 5, candidates_per_method: int = 20, emb_weight: float = 0.5, bm25_weight: float = 0.5, use_reranker: bool = True, company_filter: Optional[str] = None) -> List[Dict]:
        """Hybrid search combining BM25 and embedding with optional company filter
        
        Args:
            query: Search query
            top_k: Number of final results
            candidates_per_method: Number of candidates from each method
            emb_weight: Weight for embedding scores (default 0.5)
            bm25_weight: Weight for BM25 scores (default 0.5)
        """
        # Get candidates from both methods (without reranking yet)
        emb_results = self.search(query, top_k=candidates_per_method, ef_search=100, use_reranker=False, company_filter=company_filter)
        bm25_results = self.search_bm25(query, top_k=candidates_per_method, use_reranker=False, company_filter=company_filter)
        
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
        
    def load_index(self, path: str = 'faiss_hnsw.index'):
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(path)
        print(f"Index loaded from {path} with {self.index.ntotal} vectors")
    
    def answer(self, query: str, search_method: str = 'hybrid', top_k: int = 8, 
               company_filter: Optional[str] = None, auto_detect_company: bool = False) -> str:
        """Generate answer using retrieval and LLM with conversation context
        
        Args:
            query: User query
            search_method: 'hybrid', 'bm25', or 'semantic'
            top_k: Number of chunks to retrieve
            company_filter: Explicit company filter (overrides auto-detection)
            auto_detect_company: Whether to auto-detect company from query
        """
        if not self.use_generator or self.generator is None:
            raise ValueError("Generator not initialized. Set use_generator=True")
        
        # Auto-detect company if enabled and no explicit filter
        if auto_detect_company and not company_filter:
            detected_company = self.detect_company_in_query(query)
            if detected_company:
                company_filter = detected_company
                print(f"[Auto-detected company filter: {detected_company}]")
        
        if search_method == 'hybrid':
            chunks = self.search_hybrid(query, top_k=top_k, use_reranker=True, company_filter=company_filter)
        elif search_method == 'bm25':
            chunks = self.search_bm25(query, top_k=top_k, use_reranker=True, company_filter=company_filter)
        else:
            chunks = self.search(query, top_k=top_k, use_reranker=True, company_filter=company_filter)
        
        if not chunks:
            return f"No relevant information found{f' for {company_filter}' if company_filter else ''}."
        
        filter_msg = f" from {company_filter}" if company_filter else ""
        print(f"\n[Retrieved {len(chunks)} chunks{filter_msg} via {search_method}]")
        result = self.generator.generate(query, chunks)
        
        print("\nSources:")
        for i, src in enumerate(result['sources'][:5], 1):
            print(f"  {i}. {src['pdf']} - Page {src['page']} ({src.get('company', 'N/A')})")
        
        return result['answer']
    
    def clear_conversation(self):
        """Clear conversation history"""
        if self.generator:
            self.generator.clear_history()
    
    def get_indexed_pdfs(self) -> Dict[str, Dict]:
        """Get information about indexed PDFs"""
        return self.indexed_pdfs
    
    def get_companies(self) -> List[str]:
        """Get list of all unique companies in the indexed data"""
        companies = set()
        for info in self.indexed_pdfs.values():
            companies.add(info['company'])
        return sorted(list(companies))
    
    def detect_company_in_query(self, query: str) -> Optional[str]:
        """Detect company name mentioned in query
        
        Args:
            query: User query text
            
        Returns:
            Company name if detected, None otherwise
        """
        companies = self.get_companies()
        query_lower = query.lower()
        
        # Check for exact matches (case-insensitive)
        for company in companies:
            if company.lower() in query_lower:
                return company
        
        # Check for partial matches (e.g., "Epson" in "Seiko Epson")
        for company in companies:
            company_words = company.lower().split()
            for word in company_words:
                if len(word) > 3 and word in query_lower:
                    return company
        
        return None
    
    def search_with_auto_filter(self, query: str, top_k: int = 5, 
                                search_method: str = 'hybrid',
                                use_reranker: bool = True) -> List[Dict]:
        """Search with automatic company detection from query
        
        Args:
            query: User query
            top_k: Number of results to return
            search_method: 'hybrid', 'bm25', or 'semantic'
            use_reranker: Whether to use reranker
            
        Returns:
            List of search results with detected company info
        """
        # Try to detect company from query
        detected_company = self.detect_company_in_query(query)
        
        if detected_company:
            print(f"[Auto-detected company filter: {detected_company}]")
        
        # Perform search with detected filter
        if search_method == 'hybrid':
            results = self.search_hybrid(query, top_k=top_k, use_reranker=use_reranker, 
                                        company_filter=detected_company)
        elif search_method == 'bm25':
            results = self.search_bm25(query, top_k=top_k, use_reranker=use_reranker,
                                      company_filter=detected_company)
        else:
            results = self.search(query, top_k=top_k, use_reranker=use_reranker,
                                 company_filter=detected_company)
        
        # Add detection info to results
        for result in results:
            result['auto_filtered'] = detected_company is not None
            result['detected_company'] = detected_company
        
        return results