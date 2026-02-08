#!/usr/bin/env python3
"""RAG System Evaluation with keyword-based test cases"""

import random
from pathlib import Path
from typing import List, Dict, Optional
import json
import sys

# Add parent directory to path to import minirag
sys.path.insert(0, str(Path(__file__).parent.parent))

from minirag import RAG, get_config
from minirag.generator import Generator
from minirag.query_orchestrator import QueryOrchestrator


def calc_keyword_match(results: List[Dict], expected_keywords: List[str]) -> Dict[str, float]:
    """Calculate keyword match metrics
    
    Args:
        results: List of search result dictionaries with 'chunk' or 'content'
        expected_keywords: List of keywords that should be found
    
    Returns:
        Dictionary with match score and found keywords
    """
    if not expected_keywords:
        return {"match_score": 1.0, "found_keywords": [], "missing_keywords": []}
    
    # Combine all result text
    combined_text = " ".join([
        r.get('chunk', r.get('content', r.get('text', ''))).lower() 
        for r in results
    ])
    
    found = []
    missing = []
    for kw in expected_keywords:
        if kw.lower() in combined_text:
            found.append(kw)
        else:
            missing.append(kw)
    
    match_score = len(found) / len(expected_keywords) if expected_keywords else 0
    return {
        "match_score": match_score,
        "found_keywords": found,
        "missing_keywords": missing
    }


def calc_document_match(results: List[Dict], expected_document: Optional[str]) -> Dict[str, any]:
    """Check if results contain the expected document
    
    Args:
        results: List of search result dictionaries
        expected_document: Expected document title (can be None for cross-document queries)
    
    Returns:
        Dictionary with document match info
    """
    if not expected_document:
        return {"document_match": True, "found_documents": []}
    
    found_documents = set()
    document_match = False
    
    for r in results:
        title = r.get('title', '').lower()
        chunk = r.get('chunk', r.get('content', r.get('text', ''))).lower()
        combined = title + " " + chunk
        
        if expected_document.lower() in combined:
            document_match = True
        
        # Track which documents appear
        if title:
            found_documents.add(r.get('title', 'Unknown'))
    
    return {
        "document_match": document_match,
        "found_documents": list(found_documents)[:5]  # Top 5
    }


def evaluate(
    orchestrator: QueryOrchestrator,
    test_cases: List[Dict],
    method: str = 'hybrid',
    top_k: int = 10,
    use_reranker: bool = True,
    use_orchestrator: bool = True,
    verbose: bool = True
) -> List[Dict]:
    """Evaluate RAG system on test cases
    
    Args:
        orchestrator: QueryOrchestrator instance
        test_cases: List of test case dictionaries
        method: Search method ('embedding', 'bm25', or 'hybrid')
        top_k: Number of results to retrieve
        use_reranker: Whether to use reranker
        use_orchestrator: Whether to use LLM-based query orchestration
        verbose: Whether to print detailed output
    
    Returns:
        List of evaluation results
    """
    results = []
    
    for tc in test_cases:
        query = tc['query']
        expected_keywords = tc.get('expected_keywords', [])
        expected_document = tc.get('expected_document')
        
        # Execute query with orchestration
        if use_orchestrator:
            exec_result = orchestrator.execute_query(
                query,
                top_k=top_k,
                use_reranker=use_reranker,
                method=method,
                rephrase=True
            )
            search_results = exec_result['results']
        else:
            # Fallback to direct RAG call with simple title filter
            title_filter = expected_document if expected_document else None
            
            if method == 'hybrid':
                search_results = orchestrator.rag.search_hybrid(
                    query, 
                    top_k=top_k, 
                    use_reranker=use_reranker,
                    title_filter=title_filter
                )
            elif method == 'embedding':
                search_results = orchestrator.rag.search(
                    query, 
                    top_k=top_k, 
                    use_reranker=use_reranker,
                    title_filter=title_filter
                )
            else:  # bm25
                search_results = orchestrator.rag.search_bm25(
                    query, 
                    top_k=top_k, 
                    use_reranker=use_reranker,
                    title_filter=title_filter
                )
        
        # Calculate metrics
        keyword_metrics = calc_keyword_match(search_results, expected_keywords)
        document_metrics = calc_document_match(search_results, expected_document)
        
        # Get result info
        result_info = []
        for r in search_results[:5]:
            result_info.append({
                'title': r.get('title', 'Unknown'),
                'page': r.get('page', 0),
                'score': r.get('score', 0),
                'preview': r.get('chunk', r.get('content', ''))[:100]
            })
        
        results.append({
            'id': tc['id'],
            'query': query,
            'category': tc.get('category', 'unknown'),
            'difficulty': tc.get('difficulty', 'unknown'),
            'expected_keywords': expected_keywords,
            'expected_document': expected_document,
            'match_score': keyword_metrics['match_score'],
            'found_keywords': keyword_metrics['found_keywords'],
            'missing_keywords': keyword_metrics['missing_keywords'],
            'document_match': document_metrics['document_match'],
            'found_documents': document_metrics['found_documents'],
            'top_results': result_info,
            'num_results': len(search_results)
        })
    
    return results


def print_summary(results: List[Dict], method: str, use_reranker: bool, top_k: int):
    """Print evaluation summary with detailed metrics"""
    method_label = f"{method.upper()}" + (" + RERANKER" if use_reranker else " (BASELINE)")
    
    print(f"\n{'='*100}")
    print(f"{method_label}")
    print(f"{'='*100}")
    
    # Overall metrics
    avg_match = sum(r['match_score'] for r in results) / len(results)
    document_accuracy = sum(1 for r in results if r['document_match']) / len(results)
    
    print(f"\nOverall Metrics ({len(results)} test cases):")
    print(f"  Keyword Match Score:  {avg_match:.1%}")
    print(f"  Document Accuracy:    {document_accuracy:.1%}")
    
    # Recall and Precision @N metrics
    print(f"\nRetrieval Metrics:")
    n_values = [1, 3, 5, 10]
    
    for n in n_values:
        if n > top_k:
            continue
        
        # Calculate recall and precision at N
        recalls = []
        precisions = []
        
        for r in results:
            expected_kws = set(kw.lower() for kw in r['expected_keywords'])
            if not expected_kws:
                continue
            
            # Check top N results
            found_in_n = set()
            top_n_results = r['top_results'][:n]
            
            for res in top_n_results:
                text = (res.get('title', '') + ' ' + res.get('preview', '')).lower()
                for kw in expected_kws:
                    if kw in text:
                        found_in_n.add(kw)
            
            # Recall: what fraction of expected keywords were found
            recall = len(found_in_n) / len(expected_kws) if expected_kws else 0
            recalls.append(recall)
            
            # Precision: what fraction of retrieved results contain expected keywords
            relevant_results = 0
            for res in top_n_results:
                text = (res.get('title', '') + ' ' + res.get('preview', '')).lower()
                if any(kw in text for kw in expected_kws):
                    relevant_results += 1
            
            precision = relevant_results / n if n > 0 else 0
            precisions.append(precision)
        
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        
        print(f"  Recall@{n:2d}:    {avg_recall:.1%}")
        print(f"  Precision@{n:2d}: {avg_precision:.1%}")
    
    # By difficulty
    difficulties = {}
    for r in results:
        diff = r['difficulty']
        if diff not in difficulties:
            difficulties[diff] = []
        difficulties[diff].append(r)
    
    print(f"\nBy Difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        if diff in difficulties:
            diff_results = difficulties[diff]
            avg = sum(r['match_score'] for r in diff_results) / len(diff_results)
            print(f"  {diff:8s}: {avg:.1%} ({len(diff_results)} queries)")
    
    # By category
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    print(f"\nBy Category:")
    for cat, cat_results in sorted(categories.items(), key=lambda x: -len(x[1])):
        avg = sum(r['match_score'] for r in cat_results) / len(cat_results)
        print(f"  {cat:20s}: {avg:.1%} ({len(cat_results)} queries)")


def print_examples(results: List[Dict], n: int = 5, show_failures: bool = True):
    """Print example results"""
    if show_failures:
        # Show worst performing queries
        sorted_results = sorted(results, key=lambda x: x['match_score'])
        examples = sorted_results[:n]
        print(f"\n{'='*100}")
        print(f"LOWEST SCORING QUERIES (showing {n}):")
        print(f"{'='*100}")
    else:
        # Random sample
        examples = random.sample(results, min(n, len(results)))
        print(f"\n{'='*100}")
        print(f"SAMPLE RESULTS (showing {n} random):")
        print(f"{'='*100}")
    
    for r in examples:
        status = "✓" if r['match_score'] >= 0.5 else "✗"
        document_status = "✓" if r['document_match'] else "✗"
        
        print(f"\n{status} Test {r['id']} [{r['difficulty']}] - Match: {r['match_score']:.0%} | Document: {document_status}")
        print(f"   Query: {r['query']}")
        print(f"   Category: {r['category']}")
        if r['expected_document']:
            print(f"   Expected Document: {r['expected_document']}")
        print(f"   Keywords Found: {r['found_keywords']}")
        if r['missing_keywords']:
            print(f"   Keywords Missing: {r['missing_keywords']}")
        print(f"   Top Results:")
        for i, res in enumerate(r['top_results'][:3], 1):
            print(f"      {i}. [{res['title'][:40]}] Page {res['page']} - {res['preview'][:60]}...")


def main():
    """Main evaluation entry point"""
    import argparse
    
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).resolve().parent.parent
    
    parser = argparse.ArgumentParser(description='Evaluate RAG system with keyword-based test cases')
    parser.add_argument('--method', type=str, default='hybrid',
                       choices=['embedding', 'bm25', 'hybrid'],
                       help='Search method to evaluate')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results to retrieve')
    parser.add_argument('--no-reranker', action='store_true',
                       help='Disable reranker')
    parser.add_argument('--test-cases', type=str, default=None,
                       help='Path to test cases JSON file')
    parser.add_argument('--index', type=str, default=None,
                       help='Path to FAISS index')
    parser.add_argument('--show-failures', action='store_true',
                       help='Show lowest scoring queries instead of random sample')
    parser.add_argument('--examples', type=int, default=5,
                       help='Number of examples to show')
    parser.add_argument('--category', type=str, default=None,
                       help='Filter test cases by category')
    parser.add_argument('--difficulty', type=str, default=None,
                       choices=['easy', 'medium', 'hard'],
                       help='Filter test cases by difficulty')
    parser.add_argument('--no-orchestrator', action='store_true',
                       help='Disable LLM-based query orchestration')
    
    args = parser.parse_args()
    
    config = get_config()
    
    # Resolve paths
    test_cases_path = args.test_cases or (project_root / 'config' / 'test_cases.json')
    index_path = args.index or (project_root / config.paths.index_file)
    parsed_dir = project_root / config.paths.parsed_data_dir
    
    # Load test cases
    with open(test_cases_path, 'r') as f:
        test_cases = json.load(f)
    
    # Filter test cases if requested
    if args.category:
        test_cases = [tc for tc in test_cases if tc.get('category') == args.category]
    if args.difficulty:
        test_cases = [tc for tc in test_cases if tc.get('difficulty') == args.difficulty]
    
    if not test_cases:
        print(f"No test cases found (after filtering)")
        return
    
    print(f"Loaded {len(test_cases)} test cases from {test_cases_path}")
    
    # Load RAG
    print("\nInitializing RAG...")
    rag = RAG(config=config, use_generator=False)
    
    # Load data
    json_files = list(parsed_dir.glob("*.json"))
    if not json_files:
        print(f"No parsed JSON files found in {parsed_dir}")
        return
    
    rag.load_data([str(f) for f in json_files])
    
    # Load or build index
    index_path = Path(index_path)
    if index_path.exists():
        print(f"Loading index from {index_path}...")
        # Load index without rebuilding chunks/BM25 to check if titles are correct
        rag.load_index(str(index_path), rebuild_chunks=False, rebuild_bm25=False, rebuild_section_embeddings=False)
        
        # Verify we have chunks - if not, rebuild them
        if not rag.chunks:
            print("Rebuilding chunks and BM25 index...")
            rag.load_index(str(index_path), rebuild_chunks=True, rebuild_bm25=True, rebuild_section_embeddings=False)
        elif not rag.bm25:
            print("Rebuilding BM25 index...")
            rag._build_bm25_index()
    else:
        print(f"\n❌ ERROR: Index file not found at {index_path}")
        print("Please build the index first by running:")
        print(f"  python3 index.py")
        return
    
    # Initialize query orchestrator
    use_orchestrator = not args.no_orchestrator
    orchestrator = None
    
    if use_orchestrator:
        print("\nInitializing Query Orchestrator...")
        try:
            generator = Generator()
            orchestrator = QueryOrchestrator(rag, generator)
            print("✓ Query Orchestrator initialized (LLM-powered)")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize orchestrator: {e}")
            print("  Falling back to direct RAG calls")
            use_orchestrator = False
    
    if not use_orchestrator:
        # Create dummy orchestrator for consistent API
        class DummyOrchestrator:
            def __init__(self, rag):
                self.rag = rag
        orchestrator = DummyOrchestrator(rag)
    
    # Run evaluation
    print("\n" + "="*100)
    print("EVALUATION")
    print("="*100)
    print(f"Method: {args.method.upper()}")
    print(f"Top-k: {args.top_k}")
    print(f"Reranker: {'Disabled' if args.no_reranker else 'Enabled'}")
    print(f"Orchestration: {'Enabled (LLM)' if use_orchestrator else 'Disabled'}")
    print(f"Test cases: {len(test_cases)}")
    print("="*100)
    
    results = evaluate(
        orchestrator,
        test_cases=test_cases,
        method=args.method,
        top_k=args.top_k,
        use_reranker=not args.no_reranker,
        use_orchestrator=use_orchestrator
    )
    
    # Print summary
    print_summary(results, args.method, not args.no_reranker, args.top_k)
    
    # Print examples
    print_examples(results, n=args.examples, show_failures=args.show_failures)
    
    # Save detailed results
    results_file = project_root / 'logs' / 'evaluation_results.json'
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()

