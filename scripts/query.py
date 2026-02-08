import sys
import signal
from pathlib import Path

# Add parent directory to path to import minirag
sys.path.insert(0, str(Path(__file__).parent.parent))

from minirag import RAG, get_config
from minirag.query_orchestrator import QueryOrchestrator

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
    index_path = project_root / config.paths.index_file
    
    # Check if index exists first
    if not index_path.exists():
        print(f"\n❌ Index not found at {index_path}")
        print("Please build the index first by running:")
        print("  python3 index.py")
        return
    
    json_files = list(output_dir.glob("*.json"))
    if not json_files:
        print(f"❌ No parsed JSON files found in {output_dir}/")
        return
    
    print("="*80)
    print("MiniRAG - Interactive Query System")
    print("="*80)
    
    # Initialize RAG with generator for conversational responses
    print("Loading RAG system...")
    rag = RAG(config=config, use_generator=True)
    
    # Load all PDFs
    print(f"Loading {len(json_files)} documents...")
    rag.load_data([str(f) for f in json_files])
    
    # Load existing index (no rebuilding, no parsing)
    print(f"Loading index...")
    rag.load_index(str(index_path), rebuild_chunks=False, rebuild_bm25=False, rebuild_section_embeddings=False)
    
    # Rebuild only if needed
    if not rag.chunks:
        print("Rebuilding chunks and BM25 index...")
        rag._build_chunks()
        rag._build_bm25_index()
    elif not rag.bm25:
        print("Rebuilding BM25 index...")
        rag._build_bm25_index()
    
    # Initialize query orchestrator for smart query processing
    print("Initializing Query Orchestrator...")
    orchestrator = QueryOrchestrator(rag, rag.generator)
    
    print(f"\n✓ Ready! Loaded {len(rag.indexed_pdfs)} documents")
    
    print("\n" + "="*80)
    print("Commands:")
    print("  Just type your question - the system will automatically:")
    print("    • Extract document titles from your query")
    print("    • Rephrase for better retrieval")
    print("    • Generate conversational answers")
    print()
    print("  'exit' or 'quit' - Exit the program")
    print("  'clear' - Reset conversation history")
    print("  'titles' - Show all document titles")
    print("="*80)
    
    # Maintain conversation history for context-aware rephrasing
    conversation_history = []
    
    while True:
        query = input("\n❓ You: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
            
        if query.lower() == 'clear':
            rag.clear_conversation()
            conversation_history = []
            print("✓ Conversation cleared!")
            continue
            
        if query.lower() == 'titles':
            titles = set(info['title'] for info in rag.get_indexed_pdfs().values())
            print(f"\n📄 Document Titles ({len(titles)}):")
            for title in sorted(titles):
                print(f"  • {title}")
            continue
            
        if not query:
            continue
        
        print(f"\n{'─'*80}")
        try:
            # Use conversational orchestrator with history
            result = orchestrator.execute_conversational_query(
                query,
                conversation_history=conversation_history,
                top_k=8,
                use_reranker=True,
                method='hybrid'
            )
            
            # Show what the orchestrator did
            if result['extracted_title']:
                print(f"📄 Document: {result['extracted_title']}")
            if result['title_filter']:
                print(f"📑 Document: {result['title_filter']}")
            if result['search_query'] != query:
                print(f"🔍 Rephrased: {result['search_query']}")
            
            # Generate answer using retrieved chunks
            if result['results']:
                # Format chunks for generator
                chunks = result['results']
                
                # Show sources at top
                print(f"\n📚 Sources ({len(chunks)} retrieved):")
                for i, chunk in enumerate(chunks[:5], 1):
                    print(f"  {i}. {chunk['title']} (Page {chunk['page']})")
                if len(chunks) > 5:
                    print(f"  ... and {len(chunks)-5} more")
                
                # Generate and show answer at bottom
                answer_dict = rag.generator.generate(query, chunks)
                answer = answer_dict['answer']
                
                # Add to conversation history
                conversation_history.append((query, answer))
                
                print(f"\n{'─'*80}")
                print(f"💬 Answer:\n")
                print(answer)
            else:
                print("\n⚠️  No relevant documents found for your query.")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print('─'*80)

if __name__ == "__main__":
    main()
