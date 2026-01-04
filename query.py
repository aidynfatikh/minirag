from rag import RAG
from pathlib import Path

def main():
    output_dir = Path("parsed_data")
    data_file = list(output_dir.glob("*.json"))[0]
    
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
    
    rag.load_data(str(data_file))
    rag.build_index(chunk_size=256, overlap=32)
    
    print("\n" + "="*80)
    print("Ready! Conversational RAG (remembers context)")
    print("Commands: 'exit' to quit, 'clear' to reset conversation")
    print("="*80)
    
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            break
        if query.lower() == 'clear':
            rag.clear_conversation()
            print("Conversation cleared!")
            continue
        if not query:
            continue
        
        print(f"\n{'='*80}")
        try:
            answer = rag.answer(query, search_method='hybrid', top_k=8)
            print(f"\nAssistant: {answer}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        print('='*80)

if __name__ == "__main__":
    main()
