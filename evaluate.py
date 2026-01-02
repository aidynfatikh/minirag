#!/usr/bin/env python3
"""Simple RAG System Evaluation"""

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

def evaluate(rag, method='embedding', top_k=5):
    results = []
    for tc in TEST_CASES:
        search_results = rag.search(tc['query'], top_k, 50) if method == 'embedding' else rag.search_bm25(tc['query'], top_k)
        retrieved = [r['page'] for r in search_results]
        metrics = calc_metrics(retrieved, tc['pages'])
        results.append({**tc, 'retrieved': retrieved, **metrics})
    
    avg_p = sum(r['precision'] for r in results) / len(results)
    avg_r = sum(r['recall'] for r in results) / len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    
    print(f"\n{'='*80}\n{method.upper()} RESULTS (top_k={top_k})\n{'='*80}")
    print(f"Precision: {avg_p:.3f} | Recall: {avg_r:.3f} | F1: {avg_f1:.3f}\n{'-'*80}")
    
    for r in sorted(results, key=lambda x: x['f1']):
        status = "✓" if r['f1'] > 0.7 else "✗"
        print(f"{status} [{r['id']:2d}] F1:{r['f1']:.2f} P:{r['precision']:.2f} R:{r['recall']:.2f} | {r['query'][:50]}")
    
    return results

def main():
    print("Loading RAG system...")
    rag = RAG(model_name='all-MiniLM-L12-v2', m=32, ef_construction=256)
    rag.load_data("parsed_data/6d76ccb75bbf1b27ca60b8419c5343ac050cebb0.json")
    rag.build_index(chunk_size=512, overlap=64)
    
    emb_results = evaluate(rag, 'embedding', 5)
    bm25_results = evaluate(rag, 'bm25', 5)
    
    print(f"\n{'='*80}\nCOMPARISON\n{'='*80}")
    e_avg = sum(r['f1'] for r in emb_results) / len(emb_results)
    b_avg = sum(r['f1'] for r in bm25_results) / len(bm25_results)
    print(f"Embedding F1: {e_avg:.3f}")
    print(f"BM25 F1:      {b_avg:.3f}")
    print(f"Winner:       {'Embedding' if e_avg > b_avg else 'BM25'}\n{'='*80}")

if __name__ == "__main__":
    main()
