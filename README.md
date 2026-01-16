# MiniRAG - Lightweight RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with hybrid search, reranking, and conversational capabilities.

## Features

- **Hybrid Search**: Combines dense embeddings (E5) + sparse retrieval (BM25)
- **Cross-Encoder Reranking**: ms-marco-MiniLM for result refinement
- **Section-Aware Retrieval**: Semantic boost based on document structure
- **Conversational Generation**: LLaMA 3.2 with history tracking
- **PDF Processing**: Automatic hierarchy extraction and header detection
- **Company Filtering**: Auto-detect and filter by company
- **Evaluation Framework**: Comprehensive metrics (Recall@k, F1, precision)
- **Configuration-Driven**: All parameters in YAML config with model alternatives
- **Modular Structure**: Clean folder organization for maintainability

## Project Structure

```
minirag/
├── minirag/              # Core library (importable package)
│   ├── __init__.py       # Package exports
│   ├── rag.py            # Main RAG system
│   ├── generator.py      # LLM generation
│   ├── pdf_parser.py     # PDF processing
│   └── config_loader.py  # Configuration management
├── scripts/              # Executable scripts
│   ├── index.py          # Build vector database
│   ├── query.py          # Interactive querying
│   └── evaluate.py       # Run evaluation
├── config/               # Configuration files
│   ├── config.yaml       # Main configuration (with model alternatives)
│   └── test_cases.json   # Evaluation test cases
├── tests/                # Tests and examples
│   ├── test_config.py    # Configuration tests
│   └── examples.py       # Usage examples
├── pdfs/                 # PDF documents (add your PDFs here)
├── parsed_data/          # Parsed JSON data
├── logs/                 # Application logs
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All settings are centralized in `config/config.yaml`:

- **Models**: Embedding, reranker, and generator models (with lighter alternatives)
- **Indexing**: Chunk sizes, overlap, HNSW parameters
- **Search**: Top-k, weights, boost factors
- **Generation**: Temperature, max tokens, history window
- **Paths**: Data directories, index files

### Model Alternatives

The config file includes comments with lighter model alternatives:

**Embedding Models:**
- Current: `intfloat/e5-large-v2` (335M params, ~1.3GB)
- Alternatives: `e5-base-v2` (110M, ~440MB), `e5-small-v2` (33M, ~130MB), `all-MiniLM-L6-v2` (22M, ~90MB)

**Reranker Models:**
- Current: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- Alternatives: `L-6-v2` (2x faster), `TinyBERT-L-2-v2` (4M params, very fast)

**Generator Models:**
- Current: `Llama-3.2-3B-Instruct` (quantized)
- Alternatives: `Llama-3.2-1B` (much faster), `Phi-3-mini` (3.8B), `gemma-2b-it` (2B)

### Quick Config Example

```yaml
models:
  embedding:
    name: "intfloat/e5-large-v2"
  
indexing:
  chunk_size: 256
  overlap: 32
  hnsw:
    m: 32
    ef_construction: 256

search:
  top_k: 5
  hybrid:
    embedding_weight: 0.5
    bm25_weight: 0.5
```

## Usage

### 1. Parse PDFs

Place PDFs in `pdfs/` directory. They will be parsed automatically when building the index, or you can parse manually:

```bash
python -c "from minirag import parse_pdf; parse_pdf('pdfs/your_document.pdf')"
```

Parsed output goes to `parsed_data/`.

### 2. Build Index

```bash
python scripts/index.py --force
```

Options:
- `--chunk-size`: Chunk size in words (default: from config)
- `--overlap`: Overlap in words (default: from config)
- `--model`: Override embedding model
- `--force`: Force rebuild existing index

### 3. Query System

```bash
python scripts/query.py
```

Interactive commands:
- `exit` - Quit
- `clear` - Reset conversation
- `pdfs` - Show indexed documents
- `companies` - List all companies
- `company:<name>` - Filter by company
- `auto` - Toggle auto-detect company

### 4. Evaluate

```bash
python scripts/evaluate.py --method hybrid --top-k 10
```

Options:
- `--method`: `embedding`, `bm25`, or `hybrid`
- `--top-k`: Number of results (default: 10)
- `--no-reranker`: Disable reranking
- `--test-cases`: Path to test cases JSON

## API Examples

### Using RAG Programmatically

```python
from minirag import RAG, get_config

# Initialize with config
config = get_config()
rag = RAG(config=config, use_generator=True)

# Load and index documents
rag.load_data(["parsed_data/doc1.json", "parsed_data/doc2.json"])
rag.build_index()

# Search
results = rag.search_hybrid("What is the revenue?", top_k=5)

# Generate answer
answer = rag.answer("What is the revenue?", search_method='hybrid')
print(answer)
```

### Custom Configuration

```python
from minirag import Config

# Load custom config
config = Config.from_yaml("config/my_config.yaml")

# Override specific values
rag = RAG(
    config=config,
    model_name="custom-model",  # Override embedding model
    use_generator=True
)
```

## Configuration Parameters

### Models
- `embedding.name`: Embedding model (default: intfloat/e5-large-v2)
- `reranker.name`: Cross-encoder model (default: ms-marco-MiniLM-L-12-v2)
- `generator.name`: LLM model (default: Llama-3.2-3B-Instruct)

### Indexing
- `chunk_size`: Words per chunk (default: 256)
- `overlap`: Overlap words (default: 32)
- `hnsw.m`: HNSW connectivity (default: 32)
- `hnsw.ef_construction`: Build-time search (default: 256)

### Search
- `top_k`: Results to return (default: 5)
- `retrieve_multiplier`: Candidates multiplier (default: 4)
- `hybrid.embedding_weight`: Embedding weight (default: 0.5)
- `hybrid.bm25_weight`: BM25 weight (default: 0.5)
- `section_boost_weight`: Section boost multiplier (default: 2.0)

### Generation
- `max_tokens`: Max generation tokens (default: 800)
- `temperature`: Sampling temperature (default: 0.7)
- `history_window`: Conversation turns to keep (default: 6)
- `max_context_chunks`: Chunks for context (default: 8)

## Test Cases Format

Test cases in `test_cases.json`:

```json
[
  {
    "id": 1,
    "query": "What is the company's revenue?",
    "expected_pages": [10, 11],
    "category": "financial",
    "difficulty": "easy"
  }
]
```

## Evaluation Metrics

- **Recall@k**: How many expected pages in top-k results
- **Precision**: Ratio of relevant retrieved pages
- **F1 Score**: Harmonic mean of precision and recall
- **Per-Category**: Breakdown by question type

## Performance Tips

1. **Chunk Size**: Smaller chunks (128-256) for precise retrieval
2. **HNSW Parameters**: Higher M and ef_construction for accuracy
3. **Hybrid Weights**: Adjust based on query type (factual vs semantic)
4. **Reranker**: Always use for production (improves by 10-20%)

## Logging

Logs are saved to `logs/indexing_YYYYMMDD_HHMMSS.log` with:
- Build progress
- Chunk statistics
- Company detection
- Error traces

Configure log level in `config.yaml`:
```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests to `test_cases.json`
4. Submit pull request

## Citation

```bibtex
@software{minirag2026,
  title = {MiniRAG: Lightweight RAG System},
  year = {2026},
  author = {Your Name}
}
```
