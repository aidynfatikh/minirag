from rag import RAG
from pathlib import Path
import signal
import sys

def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully"""
    print("\n\nGracefully shutting down...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    output_dir = Path("parsed_data")
    json_files = list(output_dir.glob("*.json"))
    
    if not json_files:
        print("No parsed JSON files found in parsed_data/")
        return
    
    print("="*80)
    print("Initializing RAG with Conversational Generation")
    print("="*80)
    
    rag = RAG(
        model_name='intfloat/e5-large-v2',
        m=32,
        ef_construction=256,
        use_reranker=True,
        use_generator=True
    )
    
    # Load all PDFs
    rag.load_data([str(f) for f in json_files])
    
    # Check if index exists, load it instead of rebuilding
    index_path = "vdb.index"
    if Path(index_path).exists():
        print(f"Loading existing index from {index_path}...")
        rag.load_index(index_path)
    else:
        print("Building new index...")
        rag.build_index(chunk_size=256, overlap=32)
        rag.save_index(index_path)
        print(f"Index saved to {index_path}")
    
    # Show indexed PDFs
    print("\n" + "="*80)
    print("Indexed PDFs:")
    for pdf_name, info in rag.get_indexed_pdfs().items():
        print(f"  • {pdf_name} - {info['company']} ({info['pages']} pages)")
    
    print("\n" + "="*80)
    print("Ready! Conversational RAG (remembers context)")
    print("Commands:")
    print("  'exit' - quit")
    print("  'clear' - reset conversation")
    print("  'pdfs' - show indexed PDFs")
    print("  'companies' - show all companies")
    print("  'company:<name>' - filter by company (e.g., 'company:Epson')")
    print("  'auto' - toggle auto-detect company from query")
    print("="*80)
    
    current_filter = None
    auto_detect = False  # Auto-detect company from query
    
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
                print(f"  • {pdf_name} - {info['company']} ({info['pages']} pages)")
            continue
        if query.lower() == 'companies':
            companies = rag.get_companies()
            print(f"\nIndexed Companies ({len(companies)}):")
            for company in companies:
                pdfs = [name for name, info in rag.get_indexed_pdfs().items() if info['company'] == company]
                print(f"  • {company}: {len(pdfs)} documents")
            continue
        if query.lower() == 'auto':
            auto_detect = not auto_detect
            status = "enabled" if auto_detect else "disabled"
            print(f"Auto-detect company: {status}")
            continue
        if query.lower().startswith('company:'):
            company = query.split(':', 1)[1].strip()
            # Validate company exists
            companies = set(info['company'] for info in rag.get_indexed_pdfs().values())
            if company not in companies:
                print(f"Company '{company}' not found. Available: {', '.join(sorted(companies))}")
            else:
                current_filter = company
                print(f"Filtering by company: {company}")
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
                company_filter=filter_to_use,
                auto_detect_company=auto_detect and not current_filter
            )
            print(f"\nAssistant: {answer}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        print('='*80)

if __name__ == "__main__":
    main()
