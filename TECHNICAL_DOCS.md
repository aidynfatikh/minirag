# MiniRAG - Complete Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [PDF Parsing & Header Detection](#pdf-parsing--header-detection)
3. [Text Chunking Strategy](#text-chunking-strategy)
4. [Embedding & Indexing](#embedding--indexing)
5. [Search & Retrieval](#search--retrieval)
6. [Evaluation System](#evaluation-system)

---

## System Overview

MiniRAG is a Retrieval-Augmented Generation (RAG) system that processes PDF documents and enables semantic search using two methods:
- **Embedding-based search** using sentence transformers and FAISS
- **BM25 keyword search** for lexical matching

### Architecture Flow

```
PDF Document
    ↓
[PDF Parser] → Extract text + metadata (headers, sections, tables)
    ↓
[Text Chunking] → Split into overlapping chunks
    ↓
[Embedding Generation] → Convert to dense vectors
    ↓
[FAISS Index] → Store vectors with HNSW algorithm
    ↓
[Search] → Query → Retrieve relevant chunks
```

---

## PDF Parsing & Header Detection

**File:** `pdf_parser.py`

### 1. Core Parsing Function

```python
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # PyMuPDF library
    pages_data = []
    
    for page_num, page in enumerate(doc, 1):
        text = page.get_text().strip()
        
        # Detect headers and sections
        blocks_with_metadata = detect_headers_and_sections(page, page_num)
        
        # Extract headers for this page
        headers = [...]
        
        # Store page data with metadata
        pages_data.append({
            'page': page_num,
            'text': text,
            'metadata': {...}
        })
```

**What happens:**
- Opens PDF using PyMuPDF (fitz)
- Iterates through each page
- Extracts plain text
- Calls header detection
- Stores everything with metadata

### 2. Header & Section Detection

```python
def detect_headers_and_sections(page, page_num):
    blocks = page.get_text("dict")["blocks"]
```

**Step 1: Extract Font Information**
```python
for block in blocks:
    if block.get("type") == 0:  # Text block
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                font_size = span.get("size", 0)
                font_flags = span.get("flags", 0)
                font_name = span.get("font", "")
                
                text_blocks.append({
                    'text': text,
                    'font_size': font_size,
                    'is_bold': bool(font_flags & 2**4),  # Bit flag for bold
                    'is_italic': bool(font_flags & 2**1),  # Bit flag for italic
                })
```

**Why this works:**
- PDFs store text with font properties
- Headers typically use larger fonts or bold text
- `font_flags` is a bitmask: bit 4 = bold, bit 1 = italic
- We extract these properties for every text span

**Step 2: Calculate Font Statistics**
```python
avg_font_size = sum(font_sizes) / len(font_sizes)
max_font_size = max(font_sizes)
```

We calculate the average and max font sizes to determine what's "normal" vs "header-like"

**Step 3: Classify Headers**

Three detection methods:

**Method 1: Font Size Analysis**
```python
if font_size > avg_font_size * 1.2:
    is_header = True
    if font_size >= max_font_size * 0.95:
        header_level = 1  # Largest headers (titles)
    elif font_size >= avg_font_size * 1.5:
        header_level = 2  # Medium headers (sections)
    else:
        header_level = 3  # Small headers (subsections)
```

**Method 2: Bold Text Detection**
```python
elif is_bold and len(text.split()) <= 10:
    is_header = True
    header_level = 3
```
Short bold text is likely a header

**Method 3: Pattern Matching**
```python
section_pattern = re.match(r'^(\d+\.?)+\s+[A-Z]', text)
chapter_pattern = re.match(r'^(Chapter|Section|Part|Article)\s+\d+', text, re.IGNORECASE)

if section_pattern or chapter_pattern:
    is_header = True
    header_level = 2 if section_pattern else 1
```

Detects:
- Numbered sections: "1. Introduction", "1.2.3 Details"
- Named sections: "Chapter 1", "Section A"

**Step 4: Track Current Section**
```python
if is_header:
    current_section = text  # Update section context

classified_blocks.append({
    'text': text,
    'is_header': is_header,
    'header_level': header_level,
    'section': current_section,  # Current section this text belongs to
})
```

Each text block remembers which section it belongs to.

---

## Text Chunking Strategy

**File:** `rag.py` - `_chunk_text()` method

### Why Chunking?

Documents are too long for:
1. Embedding models (512 token limit for many models)
2. Context windows in LLMs
3. Precise retrieval (we want relevant paragraphs, not entire documents)

### Chunking Algorithm

```python
def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
```

**Step 1: Split by Sentences**
```python
sentences = re.split(r'(?<=[.!?])\s+', text)
```

- Uses regex to split on sentence boundaries
- `(?<=[.!?])` = lookbehind for `.`, `!`, or `?`
- `\s+` = followed by whitespace
- Preserves sentence structure (better than character splitting)

**Step 2: Build Chunks with Word Count Tracking**
```python
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
            # Keep last 2 sentences for overlap
            current_chunk = current_chunk[-2:]
            current_length = len(' '.join(current_chunk).split())
        else:
            current_chunk = []
            current_length = 0
    
    current_chunk.append(sentence)
    current_length += sentence_length
```

**How it works:**
1. Add sentences to chunk until reaching `chunk_size` words
2. When limit reached:
   - Save current chunk
   - Start new chunk with last 2 sentences (overlap)
3. Continue until all sentences processed

**Example:**
```
Chunk 1: [Sentence 1] [Sentence 2] [Sentence 3]
Chunk 2:                [Sentence 2] [Sentence 3] [Sentence 4] [Sentence 5]
Chunk 3:                              [Sentence 4] [Sentence 5] [Sentence 6]
         ^--------------overlap--------------^
```

**Why overlap?**
- Prevents information loss at chunk boundaries
- If a key concept spans sentences 2-3, both chunks can retrieve it
- Improves recall but slightly reduces precision

---

## Embedding & Indexing

**File:** `rag.py` - `build_index()` method

### Step 1: Load Documents & Create Chunks

```python
def build_index(self, chunk_size: int = 512, overlap: int = 50):
    for doc_idx, doc in enumerate(self.documents):
        chunks = self._chunk_text(doc['content'], chunk_size, overlap)
        
        for chunk_idx, chunk_text in enumerate(chunks):
            self.chunks.append({
                'text': chunk_text,
                'doc_idx': doc_idx,
                'chunk_idx': chunk_idx,
                'page': page_num,
                'section': section,
                'headers': headers,
                'metadata': metadata
            })
```

**What's stored:**
- `text`: The actual chunk text
- `doc_idx`: Which document this came from
- `chunk_idx`: Position within document
- `page`: PDF page number
- `section`: Current section name
- `headers`: List of headers on that page
- `metadata`: Bold/italic flags, etc.

### Step 2: Generate Embeddings

```python
print("Generating embeddings...")
texts = [chunk['text'] for chunk in self.chunks]
embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
```

**What happens:**
1. Extract all chunk texts into a list
2. Pass to `SentenceTransformer.encode()`
   - Model: `all-MiniLM-L12-v2` (384 dimensions)
   - Converts text → dense vector representation
   - Similar meanings → similar vectors
3. Returns numpy array of shape `[num_chunks, 384]`

**Example:**
```python
"Epson's revenue increased" → [0.23, -0.45, 0.12, ..., 0.67]  # 384 numbers
"Financial performance grew" → [0.25, -0.43, 0.15, ..., 0.65]  # Similar vector!
```

### Step 3: Normalize for Cosine Similarity

```python
print("Normalizing embeddings for cosine similarity...")
faiss.normalize_L2(embeddings)
```

**Mathematical explanation:**

Cosine similarity formula:
```
cos(A, B) = (A · B) / (||A|| × ||B||)
```

If we normalize vectors (make ||A|| = 1):
```
cos(A, B) = A · B
```

And L2 distance of normalized vectors:
```
||A - B||² = 2 - 2(A · B) = 2(1 - cos(A,B))
```

Therefore:
```
cos(A, B) = 1 - (L2_distance² / 2)
```

**Why normalize?**
- FAISS uses L2 distance (Euclidean)
- We want cosine similarity (angle between vectors)
- Normalization makes L2 distance equivalent to cosine similarity
- Much faster than computing cosine directly

### Step 4: Build FAISS HNSW Index

```python
self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.m)
self.index.hnsw.efConstruction = self.ef_construction
self.index.add(embeddings.astype('float32'))
```

**HNSW Algorithm Explained:**

HNSW = Hierarchical Navigable Small World

**Traditional approach (slow):**
```
Query: Find nearest neighbors
→ Compare with ALL vectors
→ Sort by distance
→ Return top K
Time: O(N) where N = number of vectors
```

**HNSW approach (fast):**
```
Build a multi-layer graph:
Layer 2: [Few nodes, long-range connections]
Layer 1: [More nodes, medium-range connections]  
Layer 0: [All nodes, local connections]

Search:
1. Start at top layer
2. Navigate graph using connections
3. Zoom down to next layer
4. Repeat until bottom
5. Return neighbors

Time: O(log N) - exponentially faster!
```

**Parameters:**

1. **`m` (default: 32)**
   - Number of connections per node
   - Higher m = better recall, more memory
   - Lower m = faster, less accurate
   - 32 is a good balance

2. **`efConstruction` (default: 200)**
   - Size of dynamic candidate list during index building
   - Higher = better graph quality, slower build time
   - 200 is standard for good quality

3. **`efSearch` (set during search)**
   - Size of dynamic candidate list during search
   - Higher = better recall, slower search
   - Can tune per-query

**Memory structure:**
```
Index stores:
- All vectors: N × 384 × 4 bytes (float32)
- Graph edges: N × m × connections
- Metadata: layer info, entry point

Example: 1000 chunks
- Vectors: 1000 × 384 × 4 = ~1.5 MB
- Graph: 1000 × 32 × 8 = ~256 KB
- Total: ~2 MB
```

### Step 5: Build BM25 Index

```python
tokenized_corpus = [chunk['text'].lower().split() for chunk in self.chunks]
self.bm25 = BM25Okapi(tokenized_corpus)
```

**BM25 Algorithm:**

BM25 = Best Matching 25 (ranking function)

**Formula:**
```
score(D, Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))

Where:
- D = document
- Q = query  
- qi = query term i
- f(qi,D) = frequency of qi in D
- |D| = length of D
- avgdl = average document length
- k1 = term frequency saturation (default: 1.5)
- b = length normalization (default: 0.75)
```

**IDF (Inverse Document Frequency):**
```python
IDF(term) = log((N - df(term) + 0.5) / (df(term) + 0.5))

Where:
- N = total documents
- df(term) = documents containing term
```

**Why this works:**
- Rare terms get higher weight (high IDF)
- Common terms get lower weight (low IDF)
- Term frequency saturates (k1 parameter)
- Normalizes for document length (b parameter)

**Example:**
```
Query: "financial report"
Doc A: "financial report financial performance" (length: 4)
Doc B: "the annual financial report was published" (length: 6)

- "financial" appears 2x in A, 1x in B
- "report" appears 1x in both
- BM25 will score based on:
  - Term frequencies
  - Document lengths
  - How rare each term is in corpus
```

---

## Search & Retrieval

### Embedding-based Search

**File:** `rag.py` - `search()` method

```python
def search(self, query: str, top_k: int = 5, ef_search: int = 50) -> List[Dict]:
```

**Step 1: Set Search Parameter**
```python
self.index.hnsw.efSearch = ef_search
```

**efSearch parameter:**
- Controls search quality vs speed tradeoff
- Higher value = more candidates explored = better recall
- Lower value = fewer candidates = faster but may miss results
- Default 50 is good balance

**Step 2: Generate Query Embedding**
```python
query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
faiss.normalize_L2(query_embedding)
```

Same process as document embeddings:
1. Convert query text to vector
2. Normalize for cosine similarity
3. Now we have a 384-dim vector representing the query

**Step 3: FAISS Search**
```python
distances, indices = self.index.search(query_embedding, top_k)
```

**What FAISS does internally:**

```
1. Start at entry point in top layer
2. Greedy search: move to nearest neighbor
3. When stuck in local minimum, go down a layer
4. Repeat until bottom layer
5. Maintain candidate list of size efSearch
6. Return top_k closest vectors

Returns:
- distances: L2 distances [0.12, 0.23, 0.45, ...]
- indices: positions in index [42, 156, 89, ...]
```

**Step 4: Convert Distance to Similarity**
```python
for dist, idx in zip(distances[0], indices[0]):
    chunk = self.chunks[idx]
    cosine_similarity = 1 - (dist ** 2) / 2
```

Recall our normalized vectors:
```
L2² = 2(1 - cosine)
cosine = 1 - (L2² / 2)
```

So we convert L2 distance back to cosine similarity (0-1 scale):
- 1.0 = identical
- 0.8+ = very similar
- 0.5-0.8 = somewhat similar
- <0.5 = different

**Step 5: Format Results**
```python
result = {
    'rank': i + 1,
    'cosine_similarity': float(cosine_similarity),
    'pdf_name': chunk['pdf_name'],
    'page': chunk['page'],
    'chunk': chunk['text'][:500],
    'section': chunk.get('section'),
    'headers': chunk.get('headers', [])
}
```

Returns all metadata so you know:
- How relevant (similarity score)
- Where it came from (page, section)
- What headers are on that page
- The actual text

### BM25 Search

**File:** `rag.py` - `search_bm25()` method

```python
def search_bm25(self, query: str, top_k: int = 5) -> List[Dict]:
```

**Step 1: Tokenize Query**
```python
tokenized_query = query.lower().split()
```

Simple whitespace tokenization:
- "What is Epson's revenue?" → ["what", "is", "epson's", "revenue?"]
- Lowercase for case-insensitive matching

**Step 2: Calculate BM25 Scores**
```python
scores = self.bm25.get_scores(tokenized_query)
```

**What happens internally:**

For each document:
```python
score = 0
for term in query:
    if term in document:
        # Calculate IDF
        idf = log((N - df + 0.5) / (df + 0.5))
        
        # Calculate term frequency component
        tf = freq[term]
        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len/avg_len))
        
        # Add to score
        score += idf * tf_component
```

Returns array of scores, one per chunk.

**Step 3: Sort and Get Top K**
```python
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
```

- Create indices [0, 1, 2, ..., N-1]
- Sort by score (descending)
- Take top K indices

**Step 4: Format Results**
```python
for idx in top_indices:
    chunk = self.chunks[idx]
    result = {
        'rank': i + 1,
        'bm25_score': float(scores[idx]),
        'page': chunk['page'],
        'chunk': chunk['text'][:500],
        ...
    }
```

### Comparison: Embedding vs BM25

| Aspect | Embedding Search | BM25 Search |
|--------|-----------------|-------------|
| **Matching Type** | Semantic (meaning) | Lexical (keywords) |
| **Example** | "revenue growth" matches "financial performance increased" | Only matches exact words |
| **Strengths** | Understands synonyms, paraphrasing | Fast, explainable, works for rare terms |
| **Weaknesses** | May miss exact matches, slower | No semantic understanding |
| **Best For** | Conceptual queries, natural language | Specific terms, technical jargon |
| **Speed** | ~5-10ms (HNSW) | ~1-2ms (direct scoring) |

**Example:**

Query: "How did COVID affect business?"

**Embedding search finds:**
- "pandemic impact on operations" ✓
- "coronavirus disrupted supply chains" ✓
- "lockdowns affected revenue" ✓

**BM25 search finds:**
- Documents containing "COVID", "business" (exact words)
- Misses: "pandemic", "coronavirus" (synonyms)

**Best practice:** Use both and combine scores (hybrid search)!

---

## Evaluation System

**File:** `evaluate.py`

### Test Cases

```python
TEST_CASES = [
    {"id": 1, "query": "What is Epson's environmental vision for 2050?", "pages": [34, 35]},
    {"id": 2, "query": "What were Epson's revenue...", "pages": [6, 44, 45]},
    ...
]
```

**Ground truth:**
- Manually labeled relevant pages for each query
- Gold standard for evaluation
- Allows objective measurement

### Metrics

**Precision:**
```python
precision = true_positives / (true_positives + false_positives)
```

**Example:**
- Retrieved pages: [34, 35, 36, 50, 60]
- Relevant pages: [34, 35, 40]
- True positives: 2 (pages 34, 35)
- False positives: 3 (pages 36, 50, 60)
- Precision = 2/5 = 0.40 (40% of results were relevant)

**Interpretation:**
- High precision = most results are relevant
- Low precision = many irrelevant results

**Recall:**
```python
recall = true_positives / (true_positives + false_negatives)
```

**Example:**
- Retrieved pages: [34, 35, 36, 50, 60]
- Relevant pages: [34, 35, 40]
- True positives: 2 (pages 34, 35)
- False negatives: 1 (page 40 not retrieved)
- Recall = 2/3 = 0.67 (67% of relevant pages found)

**Interpretation:**
- High recall = found most relevant pages
- Low recall = missed many relevant pages

**F1 Score:**
```python
f1 = 2 * (precision * recall) / (precision + recall)
```

Harmonic mean of precision and recall:
- Balances both metrics
- Penalizes when one is much lower than other
- 1.0 = perfect, 0.0 = worst

**Example:**
- Precision = 0.40, Recall = 0.67
- F1 = 2 × (0.40 × 0.67) / (0.40 + 0.67) = 0.50

### Evaluation Process

```python
def evaluate(rag, method='embedding', top_k=5):
    results = []
    for tc in TEST_CASES:
        # Search
        search_results = rag.search(tc['query'], top_k) if method == 'embedding' else rag.search_bm25(tc['query'], top_k)
        
        # Get retrieved pages
        retrieved = [r['page'] for r in search_results]
        
        # Calculate metrics
        metrics = calc_metrics(retrieved, tc['pages'])
        results.append(metrics)
```

**For each test case:**
1. Run search (embedding or BM25)
2. Extract page numbers from results
3. Compare with ground truth pages
4. Calculate precision, recall, F1
5. Store results

**Aggregate metrics:**
```python
avg_precision = sum(r['precision'] for r in results) / len(results)
avg_recall = sum(r['recall'] for r in results) / len(results)
avg_f1 = sum(r['f1'] for r in results) / len(results)
```

### Output Format

```
================================================================================
EMBEDDING RESULTS (top_k=5)
================================================================================
Precision: 0.723 | Recall: 0.651 | F1: 0.685
--------------------------------------------------------------------------------
✗ [12] F1:0.33 P:0.40 R:0.29 | How did COVID-19 impact Epson's business...
✗ [19] F1:0.50 P:0.60 R:0.43 | What is Epson's approach to climate change...
✓ [06] F1:0.80 P:1.00 R:0.67 | What is Epson's dividend policy?
✓ [10] F1:0.86 P:0.83 R:0.89 | Who are Epson's major shareholders?
✓ [11] F1:1.00 P:1.00 R:1.00 | What is the history and milestones...
```

**Interpretation:**
- ✓ = Good performance (F1 > 0.7)
- ✗ = Poor performance (F1 ≤ 0.7)
- Shows which queries work well
- Identifies areas for improvement

### Comparison

```
COMPARISON
================================================================================
Embedding F1: 0.685
BM25 F1:      0.612
Winner:       Embedding
```

Shows which method works better overall for this document and query set.

---

## Performance Characteristics

### Speed

| Operation | Time | Scale |
|-----------|------|-------|
| Parse PDF (181 pages) | ~5s | O(pages) |
| Generate embeddings (1000 chunks) | ~10s | O(chunks) |
| Build FAISS index | ~100ms | O(N log N) |
| Single search (embedding) | ~5ms | O(log N) |
| Single search (BM25) | ~2ms | O(N) but fast |

### Memory

| Component | Size (1000 chunks) |
|-----------|-------------------|
| FAISS index | ~2 MB |
| BM25 index | ~1 MB |
| Chunk metadata | ~500 KB |
| Total | ~3.5 MB |

### Scalability

**Current implementation:**
- Good for: <10,000 chunks (~50 documents)
- Memory: Fits in RAM
- Search: <10ms per query

**For larger scale:**
- Use `faiss.IndexIVFFlat` (inverted file index)
- Implement batch processing
- Use disk-based storage
- Consider distributed search (Elasticsearch)

---

## Configuration Guide

### Chunk Size Tuning

**Small chunks (256 words):**
- ✓ More precise retrieval
- ✓ Less noise in results
- ✗ May split important context
- ✗ More chunks = slower indexing

**Large chunks (1024 words):**
- ✓ More context per chunk
- ✓ Fewer chunks = faster
- ✗ Less precise
- ✗ May include irrelevant info

**Recommended:** 512 words with 64-word overlap

### HNSW Parameters

**`m` (connections per node):**
- Low (16): Fast, less accurate
- Medium (32): Balanced ✓
- High (64): Accurate, more memory

**`efConstruction` (build quality):**
- Low (100): Fast build, lower quality
- Medium (200): Balanced ✓
- High (400): Slow build, best quality

**`efSearch` (search quality):**
- Low (16): Fast, may miss results
- Medium (50): Balanced ✓
- High (200): Thorough, slower

### Model Selection

**`all-MiniLM-L6-v2`:**
- Dimensions: 384
- Speed: Fast
- Quality: Good
- Best for: Speed-critical apps

**`all-MiniLM-L12-v2`:** ✓
- Dimensions: 384
- Speed: Medium
- Quality: Better
- Best for: Balanced performance

**`all-mpnet-base-v2`:**
- Dimensions: 768
- Speed: Slower
- Quality: Best
- Best for: Quality-critical apps

---

## Common Issues & Solutions

### Issue: Low Recall

**Symptoms:** Missing relevant documents

**Solutions:**
1. Increase `top_k` (retrieve more results)
2. Increase `efSearch` (search more thoroughly)
3. Decrease chunk size (more granular chunks)
4. Use hybrid search (embedding + BM25)

### Issue: Low Precision

**Symptoms:** Many irrelevant results

**Solutions:**
1. Increase chunk size (more context per chunk)
2. Fine-tune embedding model on domain
3. Add reranking step
4. Filter by metadata (sections, headers)

### Issue: Slow Search

**Symptoms:** >50ms per query

**Solutions:**
1. Decrease `efSearch`
2. Use quantization (`IndexHNSWFlat` → `IndexHNSWPQ`)
3. Reduce embedding dimensions
4. Use GPU for embeddings

### Issue: High Memory Usage

**Symptoms:** RAM overflow

**Solutions:**
1. Increase chunk size (fewer chunks)
2. Use `IndexIVFFlat` instead of `IndexHNSWFlat`
3. Quantize embeddings (float32 → int8)
4. Process documents in batches

---

## Future Enhancements

### 1. Hybrid Search
Combine embedding + BM25 scores:
```python
final_score = alpha * embedding_score + (1 - alpha) * bm25_score
```

### 2. Reranking
Use cross-encoder to rerank top results:
```python
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, chunk) for chunk in top_results])
```

### 3. Query Expansion
Expand queries with synonyms:
```python
"revenue" → ["revenue", "income", "earnings", "sales"]
```

### 4. Metadata Filtering
Filter by section/date before search:
```python
results = search(query, filters={"section": "Financial", "year": 2022})
```

### 5. Caching
Cache frequent queries:
```python
@lru_cache(maxsize=1000)
def search_cached(query):
    return search(query)
```

---

## Summary

**PDF Parsing:**
- Extract text + font properties
- Detect headers using font size, bold, patterns
- Store metadata (sections, headers, tables)

**Chunking:**
- Split by sentences for semantic boundaries
- Use overlap to prevent information loss
- Store with metadata for filtering

**Indexing:**
- Generate 384-dim embeddings
- Normalize for cosine similarity
- Build HNSW graph for fast search
- Also build BM25 index for keywords

**Search:**
- Embedding: Semantic similarity via FAISS
- BM25: Keyword matching with TF-IDF
- Both return ranked results with scores

**Evaluation:**
- Test with 20 ground-truth queries
- Measure precision, recall, F1
- Compare methods objectively

This system provides a solid foundation for semantic search over PDF documents with sub-10ms query times and good accuracy!
