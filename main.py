from rag import RAG
from pdf_parser import process_pdf
from pathlib import Path

def main():

    pdfs_dir = Path("pdfs")
    output_dir = Path("parsed_data")
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    
    test_pdf = pdf_files[0]
    print(f"Parsing: {test_pdf.name}")
    output_path = process_pdf(test_pdf, output_dir)

    rag = RAG(model_name='all-MiniLM-L12-v2', m=32, ef_construction=256)
    rag.load_data(str(output_path))
    rag.build_index(use_summary=False, chunk_size=512, overlap=64)
    rag.save_index('vdb.index')
    
    query = input("Enter query: ")
    while query not in ["-", "exit", "quit"]:
        print(f"\nQuery: {query}")
        print("=" * 80)
        
        print("\n[BM25 RESULTS]")
        print("-" * 80)
        bm25_results = rag.search_bm25(query, top_k=5)
        for result in bm25_results:
            print(f"\nRank {result['rank']} | BM25 Score: {result['bm25_score']:.4f}")
            print(f"PDF: {result['pdf_name']} | Page: {result['page']} (Chunk {result['chunk_info']})")
            if result.get('hierarchy'):
                print(f"\n{result['hierarchy'].strip()}")
            print(f"Text: {result['chunk']}")
        
        print("\n" + "=" * 80)
        print("\n[EMBEDDING SEARCH RESULTS]")
        print("-" * 80)
        embedding_results = rag.search(query, top_k=5, ef_search=50)
        for result in embedding_results:
            print(f"\nRank {result['rank']} | Cosine Similarity: {result['cosine_similarity']:.4f}")
            print(f"PDF: {result['pdf_name']} | Page: {result['page']} (Chunk {result['chunk_info']})")
            if result.get('hierarchy'):
                print(f"\n{result['hierarchy'].strip()}")
            print(f"Text: {result['chunk']}")
        
        print("\n" + "=" * 80)
        query = input("\nEnter query: ")

if __name__ == "__main__":
    main()

