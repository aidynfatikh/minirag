#!/usr/bin/env python3
"""
Test configuration and test cases loading (no dependencies required)
"""

import sys
from pathlib import Path

# Add parent directory to path to import minirag
sys.path.insert(0, str(Path(__file__).parent.parent))

from minirag import get_config, load_test_cases, Config

def test_config_loading():
    """Test configuration loading"""
    print("="*80)
    print("Testing Configuration Loading")
    print("="*80)
    
    config = get_config()
    
    # Test model configs
    assert 'embedding' in config.models
    assert 'reranker' in config.models
    assert 'generator' in config.models
    print("✓ Model configurations present")
    
    # Test indexing config
    assert config.indexing.chunk_size > 0
    assert config.indexing.overlap >= 0
    assert config.indexing.hnsw.m > 0
    print(f"✓ Indexing config: chunk_size={config.indexing.chunk_size}, overlap={config.indexing.overlap}")
    
    # Test search config
    assert config.search.top_k > 0
    assert 0 <= config.search.hybrid.embedding_weight <= 1
    assert 0 <= config.search.hybrid.bm25_weight <= 1
    print(f"✓ Search config: top_k={config.search.top_k}")
    
    # Test generation config
    assert config.generation.max_tokens > 0
    assert 0 < config.generation.temperature <= 2
    assert config.generation.history_window > 0
    print(f"✓ Generation config: max_tokens={config.generation.max_tokens}, temp={config.generation.temperature}")
    
    # Test paths config
    assert config.paths.pdfs_dir
    assert config.paths.parsed_data_dir
    assert config.paths.index_file
    print(f"✓ Paths config: pdfs={config.paths.pdfs_dir}, parsed={config.paths.parsed_data_dir}")
    
    print("\n✓ All configuration tests passed!")
    return True


def test_test_cases_loading():
    """Test test cases loading"""
    print("\n" + "="*80)
    print("Testing Test Cases Loading")
    print("="*80)
    
    test_cases = load_test_cases()
    
    assert len(test_cases) > 0
    print(f"✓ Loaded {len(test_cases)} test cases")
    
    # Check first test case structure
    first_case = test_cases[0]
    assert 'id' in first_case
    assert 'query' in first_case
    assert 'expected_pages' in first_case
    print(f"✓ Test case structure valid")
    
    # Count categories
    categories = set(tc.get('category', 'unknown') for tc in test_cases)
    print(f"✓ Found {len(categories)} categories: {', '.join(sorted(categories))}")
    
    # Count by difficulty
    difficulties = {}
    for tc in test_cases:
        diff = tc.get('difficulty', 'unknown')
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"✓ Difficulty distribution:")
    for diff, count in sorted(difficulties.items()):
        print(f"    {diff}: {count} cases")
    
    print("\n✓ All test case tests passed!")
    return True


def test_config_values():
    """Test specific configuration values"""
    print("\n" + "="*80)
    print("Testing Configuration Values")
    print("="*80)
    
    config = get_config()
    
    # Print all configurations
    print("\n📋 Current Configuration:")
    
    print("\n  Models:")
    for model_type, model_config in config.models.items():
        print(f"    - {model_type}: {model_config.get('name', 'N/A')}")
    
    print("\n  Indexing:")
    print(f"    - Chunk size: {config.indexing.chunk_size} words")
    print(f"    - Overlap: {config.indexing.overlap} words")
    print(f"    - HNSW M: {config.indexing.hnsw.m}")
    print(f"    - HNSW ef_construction: {config.indexing.hnsw.ef_construction}")
    print(f"    - HNSW ef_search: {config.indexing.hnsw.ef_search}")
    
    print("\n  Search:")
    print(f"    - Top-k: {config.search.top_k}")
    print(f"    - Retrieve multiplier: {config.search.retrieve_multiplier}")
    print(f"    - Embedding weight: {config.search.hybrid.embedding_weight}")
    print(f"    - BM25 weight: {config.search.hybrid.bm25_weight}")
    print(f"    - Section boost weight: {config.search.section_boost_weight}")
    
    print("\n  Generation:")
    print(f"    - Max tokens: {config.generation.max_tokens}")
    print(f"    - Temperature: {config.generation.temperature}")
    print(f"    - Top-p: {config.generation.top_p}")
    print(f"    - Repetition penalty: {config.generation.repetition_penalty}")
    print(f"    - History window: {config.generation.history_window}")
    print(f"    - Max context chunks: {config.generation.max_context_chunks}")
    
    print("\n  Document Processing:")
    print(f"    - Font size multiplier: {config.document_processing.header_detection.font_size_multiplier}")
    print(f"    - Max header words: {config.document_processing.header_detection.max_header_words}")
    print(f"    - Min chunk ratio: {config.document_processing.chunking.min_chunk_ratio}")
    print(f"    - Max chunk ratio: {config.document_processing.chunking.max_chunk_ratio}")
    
    print("\n  Company Detection:")
    print(f"    - Keywords: {len(config.company_detection.keywords)} keywords")
    print(f"    - Preview lines: {config.company_detection.preview_lines}")
    
    print("\n  Paths:")
    print(f"    - PDFs: {config.paths.pdfs_dir}")
    print(f"    - Parsed data: {config.paths.parsed_data_dir}")
    print(f"    - Index file: {config.paths.index_file}")
    print(f"    - Logs: {config.paths.logs_dir}")
    
    print("\n  Logging:")
    print(f"    - Level: {config.logging.level}")
    
    print("\n✓ Configuration display complete!")
    return True


def test_file_structure():
    """Test that required files exist"""
    print("\n" + "="*80)
    print("Testing File Structure")
    print("="*80)
    
    base_path = Path(__file__).parent.parent
    
    required_files = {
        'config/config.yaml': 'Configuration file',
        'config/test_cases.json': 'Test cases',
        'requirements.txt': 'Python dependencies',
        'README.md': 'Documentation',
        'minirag/rag.py': 'Core RAG module',
        'minirag/generator.py': 'Generator module',
        'minirag/pdf_parser.py': 'PDF parser',
        'minirag/config_loader.py': 'Config loader',
        'scripts/index.py': 'Indexing script',
        'scripts/query.py': 'Query script',
        'scripts/evaluate.py': 'Evaluation script',
    }
    
    missing = []
    for filename, description in required_files.items():
        filepath = base_path / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (missing) - {description}")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠ Missing {len(missing)} files")
        return False
    else:
        print("\n✓ All required files present!")
        return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MiniRAG Configuration & Setup Tests")
    print("="*80)
    
    try:
        results = []
        results.append(("Config Loading", test_config_loading()))
        results.append(("Test Cases Loading", test_test_cases_loading()))
        results.append(("Config Values", test_config_values()))
        results.append(("File Structure", test_file_structure()))
        
        print("\n" + "="*80)
        print("Test Summary")
        print("="*80)
        
        for test_name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} - {test_name}")
        
        all_passed = all(result[1] for result in results)
        
        if all_passed:
            print("\n🎉 All tests passed! Configuration system is working correctly.")
            print("\nNext steps:")
            print("  1. Install dependencies: pip install -r requirements.txt")
            print("  2. Add PDFs to pdfs/ directory")
            print("  3. Run: python scripts/index.py")
            print("  4. Run: python scripts/query.py")
        else:
            print("\n⚠ Some tests failed. Please check the output above.")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
