import sys
import signal
from pathlib import Path

# Add parent directory to path to import minirag
sys.path.insert(0, str(Path(__file__).parent.parent))

from minirag import RAG, get_config

def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully"""
    print("\n\nGracefully shutting down...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).resolve().parent.parent
    
    config = get_config()
    output_dir = project_root / config.paths.parsed_data_dir
    json_files = list(output_dir.glob("*.json"))
    
    if not json_files:
        print(f"No parsed JSON files found in {output_dir}/")
        return
    
    print("="*80)
    print("Initializing RAG with Conversational Generation")
    print("="*80)
    
    rag = RAG(config=config, use_generator=True)
    
    # Load all PDFs
    rag.load_data([str(f) for f in json_files])
    
    # Check if index exists, load it instead of rebuilding
    index_path = project_root / config.paths.index_file
    if index_path.exists():
        print(f"Loading existing index from {index_path}...")
        rag.load_index(str(index_path), rebuild_chunks=True, rebuild_bm25=True, rebuild_section_embeddings=False)
    else:
        print("Building new index...")
        rag.build_index()
        rag.save_index(str(index_path))
        print(f"Index saved to {index_path}")
    
    # Show indexed PDFs
    print("\n" + "="*80)
    print("Indexed PDFs:")
    for pdf_name, info in rag.get_indexed_pdfs().items():
        print(f"  • {pdf_name} - {info['title']} ({info['pages']} pages)")
    
    print("\n" + "="*80)
    print("Ready! Conversational RAG (remembers context)")
    print("Commands:")
    print("  'exit' - quit")
    print("  'clear' - reset conversation")
    print("  'pdfs' - show indexed PDFs")
    print("  'titles' - show all document titles")
    print("  'title:<name>' - filter by document title (e.g., 'title:2022 Annual Report')")
    print("  'auto' - toggle auto-detect title from query")
    print("="*80)
    
    current_filter = None
    auto_detect = False  # Auto-detect title from query
    
    while True:
        prompt_prefix = f"[{current_filter}] " if current_filter else ""
        auto_prefix = "[AUTO] " if auto_detect else ""
        query = input(f"\n{auto_prefix}{prompt_prefix}You: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        if query.lower() == 'clear':
            rag.clear_conversation()
            current_filter = None
            print("Conversation cleared!")
            continue
        if query.lower() == 'pdfs':
            print("\nIndexed PDFs:")
            for pdf_name, info in rag.get_indexed_pdfs().items():
                print(f"  • {pdf_name} - {info['title']} ({info['pages']} pages)")
            continue
        if query.lower() == 'titles':
            titles = rag.get_titles()
            print(f"\nIndexed Document Titles ({len(titles)}):")
            for title in titles:
                pdfs = [name for name, info in rag.get_indexed_pdfs().items() if info['title'] == title]
                print(f"  • {title}: {len(pdfs)} documents")
            continue
        if query.lower() == 'auto':
            auto_detect = not auto_detect
            status = "enabled" if auto_detect else "disabled"
            print(f"Auto-detect title: {status}")
            continue
        if query.lower().startswith('title:'):
            title = query.split(':', 1)[1].strip()
            # Validate title exists
            titles = set(info['title'] for info in rag.get_indexed_pdfs().values())
            if title not in titles:
                print(f"Title '{title}' not found. Available: {', '.join(sorted(titles))}")
            else:
                current_filter = title
                print(f"Filtering by title: {title}")
            continue
        if not query:
            continue
        
        print(f"\n{'='*80}")
        try:
            # Use explicit filter if set, otherwise use auto-detect
            filter_to_use = current_filter if current_filter else None
            answer = rag.answer(
                query, 
                search_method='hybrid', 
                top_k=8, 
                title_filter=filter_to_use,
                auto_detect_title=auto_detect and not current_filter
            )
            print(f"\nAssistant: {answer}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        print('='*80)

if __name__ == "__main__":
    main()
