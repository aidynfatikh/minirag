#!/usr/bin/env python3
"""
Example usage of MiniRAG with configuration management
"""

from pathlib import Path
from minirag import RAG, get_config, Config

def example_basic_usage():
    """Example 1: Basic usage with default config"""
    print("="*80)
    print("Example 1: Basic Usage with Default Config")
    print("="*80)
    
    # Load default configuration
    config = get_config()
    print(f"Loaded config:")
    print(f"  - Embedding model: {config.models['embedding']['name']}")
    print(f"  - Chunk size: {config.indexing.chunk_size}")
    print(f"  - Top-k: {config.search.top_k}")
    
    # Initialize RAG with default config
    rag = RAG(config=config, use_generator=False)
    print("\n✓ RAG initialized with default config")


def example_override_params():
    """Example 2: Override specific parameters"""
    print("\n" + "="*80)
    print("Example 2: Override Specific Parameters")
    print("="*80)
    
    # Use config defaults but override specific values
    rag = RAG(
        model_name="intfloat/e5-base-v2",  # Override model
        m=16,  # Override HNSW M
        use_generator=False
    )
    print("✓ RAG initialized with custom model and HNSW M=16")


def example_custom_config():
    """Example 3: Use custom configuration file"""
    print("\n" + "="*80)
    print("Example 3: Custom Configuration File")
    print("="*80)
    
    # You can create a custom config file and load it
    # config = Config.from_yaml('my_custom_config.yaml')
    # rag = RAG(config=config)
    
    print("To use custom config:")
    print("  1. Copy config.yaml to my_config.yaml")
    print("  2. Modify settings as needed")
    print("  3. Load with: Config.from_yaml('my_config.yaml')")


def example_search_and_retrieve():
    """Example 4: Search and retrieval"""
    print("\n" + "="*80)
    print("Example 4: Search and Retrieval (requires indexed data)")
    print("="*80)
    
    config = get_config()
    
    # Check if we have data
    parsed_dir = Path(config.paths.parsed_data_dir)
    json_files = list(parsed_dir.glob("*.json"))
    
    if not json_files:
        print("⚠ No parsed data found. Run pdf_parser.py first.")
        return
    
    print(f"Found {len(json_files)} parsed documents")
    
    # Initialize and load
    rag = RAG(config=config, use_generator=False)
    rag.load_data([str(f) for f in json_files[:1]])  # Load just one for demo
    
    # Check for index
    index_path = Path(config.paths.index_file)
    if not index_path.exists():
        print("Building index...")
        rag.build_index()
        rag.save_index(str(index_path))
    else:
        print("Loading existing index...")
        rag.load_index(str(index_path))
    
    # Perform search
    print("\nSearching: 'What is the company vision?'")
    results = rag.search_hybrid("What is the company vision?", top_k=3)
    
    print(f"\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Page {result['page']} - {result['pdf_name']}")
        print(f"   Score: {result.get('final_score', 'N/A'):.3f}")
        print(f"   Snippet: {result['chunk'][:100]}...")


def example_programmatic_config():
    """Example 5: Programmatic config modification"""
    print("\n" + "="*80)
    print("Example 5: Programmatic Configuration")
    print("="*80)
    
    # Load config
    config = get_config()
    
    # Modify config programmatically
    original_chunk_size = config.indexing.chunk_size
    config.indexing.chunk_size = 512
    print(f"Changed chunk size: {original_chunk_size} → {config.indexing.chunk_size}")
    
    # Initialize with modified config
    rag = RAG(config=config, use_generator=False)
    print("✓ RAG initialized with modified config")
    
    # Restore original
    config.indexing.chunk_size = original_chunk_size


def example_config_access():
    """Example 6: Accessing configuration values"""
    print("\n" + "="*80)
    print("Example 6: Accessing Configuration Values")
    print("="*80)
    
    config = get_config()
    
    print("\nModel Configurations:")
    for model_type, model_config in config.models.items():
        print(f"  {model_type}: {model_config.get('name', 'N/A')}")
    
    print("\nIndexing Settings:")
    print(f"  Chunk size: {config.indexing.chunk_size}")
    print(f"  Overlap: {config.indexing.overlap}")
    print(f"  HNSW M: {config.indexing.hnsw.m}")
    print(f"  HNSW ef_construction: {config.indexing.hnsw.ef_construction}")
    
    print("\nSearch Settings:")
    print(f"  Default top-k: {config.search.top_k}")
    print(f"  Embedding weight: {config.search.hybrid.embedding_weight}")
    print(f"  BM25 weight: {config.search.hybrid.bm25_weight}")
    
    print("\nGeneration Settings:")
    print(f"  Max tokens: {config.generation.max_tokens}")
    print(f"  Temperature: {config.generation.temperature}")
    print(f"  History window: {config.generation.history_window}")
    
    print("\nPaths:")
    print(f"  PDFs: {config.paths.pdfs_dir}")
    print(f"  Parsed data: {config.paths.parsed_data_dir}")
    print(f"  Index: {config.paths.index_file}")
    print(f"  Logs: {config.paths.logs_dir}")


if __name__ == "__main__":
    try:
        example_basic_usage()
        example_override_params()
        example_custom_config()
        example_programmatic_config()
        example_config_access()
        
        # This requires data to be present
        example_search_and_retrieve()
        
        print("\n" + "="*80)
        print("✓ All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
