#!/usr/bin/env python3
"""
Index Builder for MiniRAG
Creates vector database from parsed PDFs with detailed logging
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import sys
import signal
from rag import RAG

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
def setup_logging(log_dir: Path = Path("logs")):
    """Setup logging to both file and console"""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"indexing_{timestamp}.log"
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - simpler output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file

def extract_company_from_json(json_path: Path, llm_extractor) -> str:
    """Extract company name from parsed JSON file using LLM
    
    Args:
        json_path: Path to parsed JSON file
        llm_extractor: Generator instance for LLM-based extraction
    
    Returns:
        Extracted company name
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get text from first 2 pages
        pages = data.get('pages', [])
        if not pages:
            return 'Unknown'
        
        text_parts = []
        for page in pages[:2]:
            page_text = page.get('text', '').strip()
            if page_text:
                text_parts.append(page_text)
        
        combined_text = '\n\n'.join(text_parts)
        if not combined_text:
            return 'Unknown'
        
        # Use LLM extraction
        extracted = llm_extractor.extract_company_name(combined_text)
        if extracted and extracted != 'Unknown' and len(extracted) > 1:
            return extracted
        
        return 'Unknown'
    except Exception as e:
        logging.warning(f"  Error: {e}")
        return 'Unknown'

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
    
    # Initialize LLM extractor
    llm_extractor = None
    try:
        from generator import Generator
        logging.info("\nInitializing LLM for company extraction...")
        llm_extractor = Generator(quantize=True)
        logging.info("✓ LLM ready\n")
    except Exception as e:
        logging.error(f"Failed to load LLM: {e}")
        return []
    
    file_info = []
    companies = {}
    total_pages = 0
    
    for idx, json_file in enumerate(json_files, 1):
        if _shutdown_requested:
            logging.warning("\nShutdown requested, stopping analysis...")
            if llm_extractor:
                llm_extractor.cleanup()
            break
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            pdf_name = data.get('pdf_name', json_file.name)
            num_pages = data.get('total_pages', len(data.get('pages', [])))
            
            logging.info(f"[{idx}/{len(json_files)}] Processing {pdf_name}...")
            company = extract_company_from_json(json_file, llm_extractor)
            
            info = {
                'json_path': str(json_file),
                'pdf_name': pdf_name,
                'company': company,
                'pages': num_pages
            }
            file_info.append(info)
            total_pages += num_pages
            
            # Track companies
            if company not in companies:
                companies[company] = {'count': 0, 'pages': 0, 'files': []}
            companies[company]['count'] += 1
            companies[company]['pages'] += num_pages
            companies[company]['files'].append(pdf_name)
            
            logging.info(f"  ✓ Company: {company}")
            
        except Exception as e:
            logging.error(f"  ✗ Error reading {json_file.name}: {e}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Total: {len(file_info)} PDFs, {total_pages} pages")
    logging.info(f"\nCompanies detected:")
    for company, info in sorted(companies.items(), key=lambda x: x[1]['count'], reverse=True):
        logging.info(f"  • {company}: {info['count']} documents, {info['pages']} pages")
    
    # Show warning if too many unknowns
    unknown_count = companies.get('Unknown', {}).get('count', 0)
    if unknown_count > len(json_files) * 0.3:
        logging.warning(f"\n⚠ Warning: {unknown_count} documents with unknown company ({unknown_count*100//len(json_files)}%)")
        logging.warning("  Consider reviewing the first pages of these documents manually.")
    
    # Cleanup LLM
    if llm_extractor:
        llm_extractor.cleanup()
    
    return file_info

def build_vector_database(
    parsed_dir: Path = Path("parsed_data"),
    index_path: Path = Path("vdb.index"),
    model_name: str = 'intfloat/e5-large-v2',
    chunk_size: int = 256,
    overlap: int = 32,
    m: int = 32,
    ef_construction: int = 256,
    force_rebuild: bool = False
):
    """Build vector database from parsed PDFs with detailed logging"""
    
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
    
    # Analyze data with LLM-based company extraction
    file_info = analyze_parsed_data(parsed_dir)
    if not file_info:
        logging.error("No data to index!")
        return
    
    # Initialize RAG
    logging.info("\n" + "="*80)
    logging.info("Initializing RAG system...")
    logging.info("="*80)
    
    try:
        rag = RAG(
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
        json_paths = [info['json_path'] for info in file_info]
        rag.load_data(json_paths)
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
        companies = {}
        for chunk in rag.chunks:
            company = chunk.get('company', 'Unknown')
            if company not in companies:
                companies[company] = 0
            companies[company] += 1
        
        logging.info(f"\nChunk distribution by company:")
        for company, count in sorted(companies.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  • {company}: {count} chunks")
            
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
    logging.info(f"\nIndexed documents by company:")
    company_pdfs = {}
    for pdf_name, info in rag.indexed_pdfs.items():
        company = info['company']
        if company not in company_pdfs:
            company_pdfs[company] = []
        company_pdfs[company].append(f"{pdf_name} ({info['pages']} pages)")
    
    for company, pdfs in sorted(company_pdfs.items()):
        logging.info(f"\n  {company} ({len(pdfs)} documents):")
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
    
    parser = argparse.ArgumentParser(description='Index PDFs into vector database')
    parser.add_argument('--parsed-dir', type=str, default='parsed_data',
                       help='Directory containing parsed JSON files')
    parser.add_argument('--index-path', type=str, default='vdb.index',
                       help='Path to save FAISS index')
    parser.add_argument('--model', type=str, default='intfloat/e5-large-v2',
                       help='Embedding model name')
    parser.add_argument('--chunk-size', type=int, default=256,
                       help='Chunk size in words')
    parser.add_argument('--overlap', type=int, default=32,
                       help='Overlap size in words')
    parser.add_argument('--m', type=int, default=32,
                       help='HNSW M parameter')
    parser.add_argument('--ef-construction', type=int, default=256,
                       help='HNSW ef_construction parameter')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild even if index exists')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for log files')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(Path(args.log_dir))
    logging.info(f"Logging to: {log_file}")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Build index
        build_vector_database(
            parsed_dir=Path(args.parsed_dir),
            index_path=Path(args.index_path),
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
