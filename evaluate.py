#!/usr/bin/env python3
"""Simple RAG System Evaluation"""

import random
from rag import RAG
from pathlib import Path

TEST_CASES = [
    {"id": 1, "query": "What is Epson's environmental vision for 2050?", "pages": [34, 35]},
    {"id": 2, "query": "What were Epson's revenue and operating results in fiscal year 2022?", "pages": [6, 44, 45]},
    {"id": 3, "query": "What are the major business segments of Epson?", "pages": [9, 10, 11, 12]},
    {"id": 4, "query": "What risks does Epson face in its business operations?", "pages": [37, 38, 39, 40, 41, 42]},
    {"id": 5, "query": "Tell me about Epson's printing solutions business", "pages": [9, 10]},
    {"id": 6, "query": "What is Epson's dividend policy?", "pages": [66]},
    {"id": 7, "query": "How does Epson approach corporate governance?", "pages": [67, 68, 69, 70]},
    {"id": 8, "query": "What are Epson's research and development activities?", "pages": [51, 52]},
    {"id": 9, "query": "What major equipment and facilities does Epson have?", "pages": [54, 55]},
    {"id": 10, "query": "Who are Epson's major shareholders?", "pages": [58, 59, 60]},
    {"id": 11, "query": "What is the history and milestones of Seiko Epson Corporation?", "pages": [7, 8]},
    {"id": 12, "query": "How did COVID-19 impact Epson's business in 2022?", "pages": [44, 45]},
    {"id": 13, "query": "What are Epson's plans for capital expenditures?", "pages": [53, 56]},
    {"id": 14, "query": "Describe Epson's visual communications business segment", "pages": [10, 11]},
    {"id": 15, "query": "What is Epson 25 Renewed corporate vision?", "pages": [19, 20, 34, 44]},
    {"id": 16, "query": "How many employees does Epson have?", "pages": [18, 19]},
    {"id": 17, "query": "What are the exchange rate impacts on Epson's financial results?", "pages": [6, 44, 45]},
    {"id": 18, "query": "Tell me about Epson's subsidiaries and affiliated entities", "pages": [13, 14, 15, 16, 17]},
    {"id": 19, "query": "What is Epson's approach to climate change and TCFD?", "pages": [34, 35]},
    {"id": 20, "query": "What are the major management contracts of Epson?", "pages": [50]},
    {"id": 21, "query": "What are Epson's inventory and asset values?", "pages": [48, 49]},
    {"id": 22, "query": "Describe Epson's wearables and industrial products business", "pages": [11, 12]},
    {"id": 23, "query": "What is the board structure and executive compensation at Epson?", "pages": [71, 72, 73]},
    {"id": 24, "query": "Tell me about Epson's manufacturing segment business", "pages": [12]},
    {"id": 25, "query": "What are Epson's intellectual property and patent strategies?", "pages": [51, 52]},
    {"id": 26, "query": "How does Epson manage supply chain and logistics risks?", "pages": [37, 38, 39]},
    {"id": 27, "query": "What is Epson's debt and financing structure?", "pages": [48, 49, 56]},
    {"id": 28, "query": "Describe Epson's sales and distribution network globally", "pages": [13, 14, 15]},
    {"id": 29, "query": "What are Epson's plans for digital transformation?", "pages": [19, 20, 44]},
    {"id": 30, "query": "How does Epson address human rights and labor practices?", "pages": [34, 35]},
    {"id": 31, "query": "What are the key financial ratios and performance indicators?", "pages": [44, 45, 46]},
    {"id": 32, "query": "Tell me about Epson's microdevices and other products", "pages": [12]},
    {"id": 33, "query": "What is Epson's strategy for emerging markets?", "pages": [9, 10, 44]},
    {"id": 34, "query": "How does Epson ensure audit and internal control quality?", "pages": [67, 68, 69]},
    {"id": 35, "query": "What are the geographical revenue breakdowns?", "pages": [9, 10, 44, 45]},
    {"id": 36, "query": "Describe Epson's sustainability goals and ESG initiatives", "pages": [34, 35]},
    {"id": 37, "query": "What legal proceedings or litigation is Epson involved in?", "pages": [50]},
    {"id": 38, "query": "How does Epson approach product development and innovation?", "pages": [51, 52]},
    {"id": 39, "query": "What are the trends in Epson's operating income and margins?", "pages": [44, 45, 46]},
    {"id": 40, "query": "Tell me about Epson's treasury stock and share repurchase programs", "pages": [60, 61, 66]},
]

def calc_metrics(retrieved, expected):
    r_set, e_set = set(retrieved), set(expected)
    tp = len(r_set & e_set)
    fp = len(r_set - e_set)
    fn = len(e_set - r_set)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return {"precision": p, "recall": r, "f1": f1}

def calc_recall_at_k(retrieved, expected, k):
    """Calculate recall@k: how many expected pages found in top k results"""
    retrieved_k = set(retrieved[:k])
    expected_set = set(expected)
    hits = len(retrieved_k & expected_set)
    return hits / len(expected_set) if expected_set else 0

def evaluate(rag, method='embedding', top_k=10, use_reranker=True, candidates_per_method=50):
    results = []
    for tc in TEST_CASES:
        if method == 'hybrid':
            search_results = rag.search_hybrid(tc['query'], top_k, candidates_per_method, use_reranker)
        elif method == 'embedding':
            search_results = rag.search(tc['query'], top_k, 50, use_reranker)
        else:  # bm25
            search_results = rag.search_bm25(tc['query'], top_k, use_reranker)
            
        retrieved = [r['page'] for r in search_results]
        metrics = calc_metrics(retrieved, tc['pages'])
        
        # Calculate recall@k for k=1,3,5,10
        r_at_1 = calc_recall_at_k(retrieved, tc['pages'], 1)
        r_at_3 = calc_recall_at_k(retrieved, tc['pages'], 3)
        r_at_5 = calc_recall_at_k(retrieved, tc['pages'], 5)
        r_at_10 = calc_recall_at_k(retrieved, tc['pages'], min(10, len(retrieved)))
        
        results.append({
            **tc, 
            'retrieved': retrieved, 
            **metrics,
            'recall_at_1': r_at_1,
            'recall_at_3': r_at_3,
            'recall_at_5': r_at_5,
            'recall_at_10': r_at_10
        })
    
    # Calculate averages
    avg_p = sum(r['precision'] for r in results) / len(results)
    avg_r = sum(r['recall'] for r in results) / len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    avg_r1 = sum(r['recall_at_1'] for r in results) / len(results)
    avg_r3 = sum(r['recall_at_3'] for r in results) / len(results)
    avg_r5 = sum(r['recall_at_5'] for r in results) / len(results)
    avg_r10 = sum(r['recall_at_10'] for r in results) / len(results)
    
    method_label = f"{method.upper()}" + (" + RERANKER" if use_reranker else " (BASELINE)")
    print(f"\n{'='*100}\n{method_label}\n{'='*100}")
    print(f"Recall@1: {avg_r1:.3f} | Recall@3: {avg_r3:.3f} | Recall@5: {avg_r5:.3f} | Recall@10: {avg_r10:.3f} | F1: {avg_f1:.3f}")
    print(f"{'='*100}\n")
    print(f"Showing 5 random examples (out of {len(results)} total):")
    print("-"*100)
    
    # Show 5 random examples
    sample_results = random.sample(results, min(5, len(results)))
    for r in sample_results:
        status = "✓" if r['f1'] > 0.5 else "✗"
        print(f"{status} [{r['id']:2d}] R@1:{r['recall_at_1']:.2f} R@3:{r['recall_at_3']:.2f} R@5:{r['recall_at_5']:.2f} R@10:{r['recall_at_10']:.2f} F1:{r['f1']:.2f}")
        print(f"    Query: {r['query']}")
        print(f"    Expected pages: {r['pages']}")
        print(f"    Retrieved top 10: {r['retrieved'][:10]}")
        # Show which pages match
        matches = set(r['retrieved']) & set(r['pages'])
        if matches:
            print(f"    ✓ Matches: {sorted(matches)}")
        print()
    
    return results, {'recall_at_1': avg_r1, 'recall_at_3': avg_r3, 'recall_at_5': avg_r5, 'recall_at_10': avg_r10, 'f1': avg_f1}

def main():
    print("Loading RAG system with E5-large-v2 embeddings...")
    rag = RAG(model_name='intfloat/e5-large-v2', m=32, ef_construction=256, use_reranker=True)
    rag.load_data("parsed_data/6d76ccb75bbf1b27ca60b8419c5343ac050cebb0.json")
    rag.build_index(chunk_size=256, overlap=32)
    
    print("\n" + "="*100)
    print("EVALUATING: HYBRID SEARCH (BM25 + Embedding + Reranker)")
    print("="*100)
    
    # Evaluate hybrid approach with 50 candidates from each method
    _, hybrid_metrics = evaluate(rag, 'hybrid', top_k=10, use_reranker=True, candidates_per_method=50)
    
    # Also evaluate individual methods for comparison
    _, emb_metrics = evaluate(rag, 'embedding', top_k=10, use_reranker=True)
    _, bm25_metrics = evaluate(rag, 'bm25', top_k=10, use_reranker=True)
    
    # Final comparison table
    print(f"\n{'='*100}")
    print("FINAL COMPARISON - ALL METHODS")
    print("="*100)
    print(f"{'Method':<30} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'F1':>8}")
    print("-"*100)
    
    print(f"{'HYBRID (BM25+Emb+Rerank)':<30} {hybrid_metrics['recall_at_1']:>8.3f} {hybrid_metrics['recall_at_3']:>8.3f} {hybrid_metrics['recall_at_5']:>8.3f} {hybrid_metrics['recall_at_10']:>8.3f} {hybrid_metrics['f1']:>8.3f}")
    print(f"{'Embedding + Reranker':<30} {emb_metrics['recall_at_1']:>8.3f} {emb_metrics['recall_at_3']:>8.3f} {emb_metrics['recall_at_5']:>8.3f} {emb_metrics['recall_at_10']:>8.3f} {emb_metrics['f1']:>8.3f}")
    print(f"{'BM25 + Reranker':<30} {bm25_metrics['recall_at_1']:>8.3f} {bm25_metrics['recall_at_3']:>8.3f} {bm25_metrics['recall_at_5']:>8.3f} {bm25_metrics['recall_at_10']:>8.3f} {bm25_metrics['f1']:>8.3f}")
    
    print("="*100)

if __name__ == "__main__":
    main()
