#!/usr/bin/env python3
"""
Index Builder for MiniRAG
Creates vector database from parsed JSON files in parsed_data/ folder
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import sys
import signal

# Add parent directory to path to import minirag
sys.path.insert(0, str(Path(__file__).parent.parent))

from minirag import RAG, get_config

# Global flag for graceful shutdown
_shutdown_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully"""
    global _shutdown_requested
    if _shutdown_requested:
        print("\n\nForce quit!")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\nShutdown requested... cleaning up (press Ctrl-C again to force quit)")

# Configure logging
def setup_logging(log_dir: Optional[Path] = None, config=None) -> Path:
    """Setup logging to both file and console
    
    Args:
        log_dir: Directory for log files (uses config default if None)
        config: Configuration object (loads default if None)
    
    Returns:
        Path to created log file
    """
    if config is None:
        config = get_config()
    
    # Get project root for proper path resolution
    project_root = Path(__file__).parent.parent
    if log_dir is None:
        log_dir = project_root / config.paths.logs_dir
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime(config.logging.file_timestamp_format)
    log_file = log_dir / f"indexing_{timestamp}.log"
    
    # Setup logging format
    log_format = config.logging.format
    date_format = config.logging.date_format
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Setup root logger
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    return log_file

def analyze_parsed_data(parsed_dir: Path) -> List[Dict]:
    """Analyze all parsed JSON files and extract metadata
    
    Args:
        parsed_dir: Directory containing parsed JSON files
    
    Returns:
        List of file info dictionaries
    """
    logging.info("="*80)
    logging.info("Analyzing parsed data...")
    logging.info("="*80)
    
    json_files = list(parsed_dir.glob("*.json"))
    if not json_files:
        logging.error(f"No JSON files found in {parsed_dir}")
        return []
    
    logging.info(f"Found {len(json_files)} parsed PDF files")
    
    file_info = []
    titles = {}
    total_pages = 0
    
    for idx, json_file in enumerate(json_files, 1):
        if _shutdown_requested:
            logging.warning("\nShutdown requested, stopping analysis...")
            break
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            pdf_name = data.get('pdf_name', json_file.name)
            num_pages = data.get('total_pages', len(data.get('pages', [])))
            
            logging.info(f"[{idx}/{len(json_files)}] Processing {pdf_name}...")
            
            # Get title from metadata (should be present from parse.py)
            title = data.get('title', 'Unknown')
            if not title or title in ['Unknown', 'unknown', '']:
                logging.warning(f"  ⚠ No title in metadata, using filename")
                title = pdf_name.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
            
            info = {
                'json_path': str(json_file),
                'pdf_name': pdf_name,
                'title': title,
                'pages': num_pages
            }
            file_info.append(info)
            total_pages += num_pages
            
            # Track titles
            if title not in titles:
                titles[title] = {'count': 0, 'pages': 0, 'files': []}
            titles[title]['count'] += 1
            titles[title]['pages'] += num_pages
            titles[title]['files'].append(pdf_name)
            
            logging.info(f"  ✓ Title: {title}")
            
        except Exception as e:
            logging.error(f"  ✗ Error reading {json_file.name}: {e}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Total: {len(file_info)} PDFs, {total_pages} pages")
    logging.info(f"\nTitles detected:")
    for title, info in sorted(titles.items(), key=lambda x: x[1]['count'], reverse=True):
        logging.info(f"  • {title}: {info['count']} documents, {info['pages']} pages")
    
    # Show warning if too many unknowns
    unknown_count = titles.get('Unknown', {}).get('count', 0)
    if unknown_count > len(json_files) * 0.3:
        logging.warning(f"\n⚠ Warning: {unknown_count} documents with unknown title ({unknown_count*100//len(json_files)}%)")
        logging.warning("  Consider re-parsing these documents with: python scripts/parse.py")
    
    return file_info

def build_vector_database(
    parsed_dir: Optional[Path] = None,
    index_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    m: Optional[int] = None,
    ef_construction: Optional[int] = None,
    force_rebuild: bool = False
):
    """Build vector database from parsed PDFs with detailed logging
    
    All parameters are optional and will use config defaults if not provided.
    """
    config = get_config()
    
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).resolve().parent.parent
    
    # Use config defaults if not provided
    parsed_dir = parsed_dir or (project_root / config.paths.parsed_data_dir)
    index_path = index_path or (project_root / config.paths.index_file)
    model_name = model_name or config.models.get('embedding', {}).get('name', 'intfloat/e5-large-v2')
    chunk_size = chunk_size or config.indexing.chunk_size
    overlap = overlap or config.indexing.overlap
    m = m or config.indexing.hnsw.m
    ef_construction = ef_construction or config.indexing.hnsw.ef_construction
    
    logging.info("="*80)
    logging.info("VECTOR DATABASE INDEXING")
    logging.info("="*80)
    logging.info(f"Configuration:")
    logging.info(f"  Model: {model_name}")
    logging.info(f"  Chunk size: {chunk_size} words")
    logging.info(f"  Overlap: {overlap} words")
    logging.info(f"  HNSW M: {m}")
    logging.info(f"  HNSW ef_construction: {ef_construction}")
    logging.info(f"  Index path: {index_path}")
    logging.info("="*80)
    
    # Check if index exists
    if index_path.exists() and not force_rebuild:
        logging.warning(f"\nIndex already exists at {index_path}")
        response = input("Do you want to rebuild it? (y/N): ").strip().lower()
        if response != 'y':
            logging.info("Skipping index rebuild. Use --force to override.")
            return
    
    # Analyze data and extract titles from metadata
    file_info = analyze_parsed_data(parsed_dir)
    if not file_info:
        logging.error("No data to index!")
        logging.error("Please parse PDFs first using: python scripts/parse.py")
        return
    
    # Initialize RAG
    logging.info("\n" + "="*80)
    logging.info("Initializing RAG system...")
    logging.info("="*80)
    
    try:
        rag = RAG(
            config=config,
            model_name=model_name,
            m=m,
            ef_construction=ef_construction,
            use_reranker=True,
            use_generator=False  # Not needed for indexing
        )
        logging.info("✓ RAG system initialized")
    except Exception as e:
        logging.error(f"✗ Failed to initialize RAG: {e}")
        raise
    
    # Load data
    logging.info("\n" + "="*80)
    logging.info("Loading parsed PDFs...")
    logging.info("="*80)
    
    try:
        # Create mapping from json_path to extracted title
        json_paths = [info['json_path'] for info in file_info]
        titles_map = {info['json_path']: info['title'] for info in file_info}
        
        rag.load_data(json_paths, titles_override=titles_map)
        logging.info(f"✓ Loaded {len(rag.documents)} pages from {len(json_paths)} PDFs")
    except Exception as e:
        logging.error(f"✗ Failed to load data: {e}")
        raise
    
    # Build index
    logging.info("\n" + "="*80)
    logging.info("Building vector index...")
    logging.info("="*80)
    
    try:
        rag.build_index(chunk_size=chunk_size, overlap=overlap)
        logging.info(f"✓ Index built with {len(rag.chunks)} chunks")
        
        # Show chunk statistics
        titles = {}
        for chunk in rag.chunks:
            title = chunk.get('title', 'Unknown')
            if title not in titles:
                titles[title] = 0
            titles[title] += 1
        
        logging.info(f"\nChunk distribution by title:")
        for title, count in sorted(titles.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  • {title}: {count} chunks")
            
    except Exception as e:
        logging.error(f"✗ Failed to build index: {e}")
        raise
    
    # Save index
    logging.info("\n" + "="*80)
    logging.info("Saving index to disk...")
    logging.info("="*80)
    
    try:
        rag.save_index(str(index_path))
        logging.info(f"✓ Index saved to {index_path}")
        
        # Show file size
        size_mb = index_path.stat().st_size / (1024 * 1024)
        logging.info(f"  Index size: {size_mb:.2f} MB")
        
    except Exception as e:
        logging.error(f"✗ Failed to save index: {e}")
        raise
    
    # Summary
    logging.info("\n" + "="*80)
    logging.info("INDEXING COMPLETE")
    logging.info("="*80)
    logging.info(f"Indexed PDFs: {len(rag.indexed_pdfs)}")
    logging.info(f"Total pages: {len(rag.documents)}")
    logging.info(f"Total chunks: {len(rag.chunks)}")
    logging.info(f"Embedding dimension: {rag.embedding_dim}")
    logging.info(f"Index vectors: {rag.index.ntotal}")
    
    # Show indexed PDFs
    logging.info(f"\nIndexed documents by title:")
    title_pdfs = {}
    for pdf_name, info in rag.indexed_pdfs.items():
        title = info['title']
        if title not in title_pdfs:
            title_pdfs[title] = []
        title_pdfs[title].append(f"{pdf_name} ({info['pages']} pages)")
    
    for title, pdfs in sorted(title_pdfs.items()):
        logging.info(f"\n  {title} ({len(pdfs)} documents):")
        for pdf in pdfs[:5]:  # Show first 5
            logging.info(f"    - {pdf}")
        if len(pdfs) > 5:
            logging.info(f"    ... and {len(pdfs) - 5} more")
    
    logging.info("\n" + "="*80)
    logging.info(f"Index ready for use!")
    logging.info("="*80)
    
    return rag

def main():
    """Main entry point"""
    import argparse
    
    # Get project root for path resolution
    project_root = Path(__file__).resolve().parent.parent
    
    parser = argparse.ArgumentParser(description='Index PDFs into vector database')
    parser.add_argument('--parsed-dir', type=str, default=None,
                       help='Directory containing parsed JSON files (default: parsed_data)')
    parser.add_argument('--index-path', type=str, default=None,
                       help='Path to save FAISS index (default: vdb.index)')
    parser.add_argument('--model', type=str, default=None,
                       help='Embedding model name')
    parser.add_argument('--chunk-size', type=int, default=None,
                       help='Chunk size in words')
    parser.add_argument('--overlap', type=int, default=None,
                       help='Overlap size in words')
    parser.add_argument('--m', type=int, default=None,
                       help='HNSW M parameter')
    parser.add_argument('--ef-construction', type=int, default=None,
                       help='HNSW ef_construction parameter')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild even if index exists')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for log files (default: logs)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    log_dir = Path(args.log_dir) if args.log_dir else project_root / 'logs'
    if args.log_dir and not Path(args.log_dir).is_absolute():
        log_dir = project_root / args.log_dir
    
    parsed_dir = None
    if args.parsed_dir:
        parsed_dir = Path(args.parsed_dir)
        if not parsed_dir.is_absolute():
            parsed_dir = project_root / args.parsed_dir
    
    index_path = None
    if args.index_path:
        index_path = Path(args.index_path)
        if not index_path.is_absolute():
            index_path = project_root / args.index_path
    
    # Setup logging
    log_file = setup_logging(log_dir)
    logging.info(f"Logging to: {log_file}")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Build index
        build_vector_database(
            parsed_dir=parsed_dir,
            index_path=index_path,
            model_name=args.model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            m=args.m,
            ef_construction=args.ef_construction,
            force_rebuild=args.force
        )
        
        logging.info("\n✓ Indexing completed successfully!")
        
    except KeyboardInterrupt:
        logging.warning("\n\n✗ Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n✗ Indexing failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
