"""
Query Orchestrator - Uses LLM to understand queries and formulate optimal RAG calls
"""
from typing import Dict, List, Optional
import json
import re
import torch


class QueryOrchestrator:
    """Orchestrates query execution by analyzing intent and formulating RAG calls"""
    
    def __init__(self, rag, generator):
        """Initialize orchestrator with RAG system and Generator (LLM)
        
        Args:
            rag: RAG system instance
            generator: Generator instance for query analysis
        """
        self.rag = rag
        self.generator = generator
        
    def _extract_company_from_query(self, query: str) -> Optional[str]:
        """Use LLM to extract company name from query
        
        Args:
            query: User query
            
        Returns:
            Company name or None
        """
        prompt_messages = [
            {
                "role": "system",
                "content": "You extract company names from queries. Respond with ONLY the company name or 'NONE'."
            },
            {
                "role": "user",
                "content": f"Query: \"{query}\"\n\nIf a specific company is mentioned, respond with ONLY the company name. If no specific company or if it mentions 'any company', 'companies', or multiple companies, respond with: NONE\n\nCompany:"
            }
        ]
        
        try:
            prompt = self.generator.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.generator.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.generator.device)
            
            with torch.no_grad():
                outputs = self.generator.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.generator.tokenizer.pad_token_id,
                )
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            company = self.generator.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            if company.upper() == "NONE" or not company:
                return None
            
            # Clean up response
            company = company.replace('"', '').replace("'", "").strip()
            return company if company else None
            
        except Exception as e:
            print(f"Warning: Failed to extract company: {e}")
            return None
    
    def _match_company_to_title(self, company_name: str) -> Optional[str]:
        """Find matching document title for a company name
        
        Uses fuzzy matching to find titles containing the company name
        
        Args:
            company_name: Company name to search for
            
        Returns:
            Matching title or None
        """
        if not company_name:
            return None
        
        # Get unique titles from indexed PDFs
        available_titles = set()
        for pdf_info in self.rag.indexed_pdfs.values():
            if 'title' in pdf_info:
                available_titles.add(pdf_info['title'])
        
        # Try exact match first
        for title in available_titles:
            if company_name.lower() in title.lower():
                return title
        
        # Try partial word matching
        company_words = set(company_name.lower().split())
        best_match = None
        best_score = 0
        
        for title in available_titles:
            title_words = set(title.lower().split())
            common_words = company_words & title_words
            score = len(common_words) / len(company_words) if company_words else 0
            
            if score > best_score and score > 0.5:  # At least 50% word overlap
                best_score = score
                best_match = title
        
        return best_match
    
    def _rephrase_query_for_retrieval(self, query: str, company_name: Optional[str]) -> str:
        """Use LLM to rephrase query for better retrieval
        
        Args:
            query: Original query
            company_name: Extracted company name
            
        Returns:
            Rephrased query optimized for search
        """
        if company_name:
            context = f"This query is about {company_name}."
        else:
            context = "This query is general or cross-company."
        
        prompt_messages = [
            {
                "role": "system",
                "content": "You rephrase queries for document retrieval. Focus on key concepts and remove question words."
            },
            {
                "role": "user",
                "content": f"{context}\n\nOriginal: \"{query}\"\n\nRephrase for search by:\n1. Removing question words\n2. Keeping key terms\n3. Expanding acronyms\n\nRephrased:"
            }
        ]
        
        try:
            prompt = self.generator.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.generator.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.generator.device)
            
            with torch.no_grad():
                outputs = self.generator.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.generator.tokenizer.pad_token_id,
                )
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            rephrased = self.generator.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            rephrased = rephrased.replace('"', '').replace("'", "")
            
            # If rephrasing failed or is too short, use original
            if len(rephrased) < 5:
                return query
            
            return rephrased
            
        except Exception as e:
            print(f"Warning: Failed to rephrase query: {e}")
            return query
    
    def execute_query(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: bool = True,
        method: str = 'hybrid',
        rephrase: bool = True
    ) -> Dict:
        """Execute query with intelligent orchestration
        
        Args:
            query: User query
            top_k: Number of results
            use_reranker: Whether to use reranker
            method: Search method ('hybrid', 'embedding', or 'bm25')
            rephrase: Whether to rephrase query
            
        Returns:
            Dictionary with results and metadata
        """
        # Extract company name
        company_name = self._extract_company_from_query(query)
        
        # Match company to actual title
        title_filter = None
        if company_name:
            title_filter = self._match_company_to_title(company_name)
            if not title_filter:
                print(f"Warning: Could not find document for company '{company_name}'")
        
        # Rephrase query if requested
        search_query = query
        if rephrase:
            search_query = self._rephrase_query_for_retrieval(query, company_name)
        
        # Execute search
        if method == 'hybrid':
            results = self.rag.search_hybrid(
                search_query,
                top_k=top_k,
                use_reranker=use_reranker,
                title_filter=title_filter
            )
        elif method == 'embedding':
            results = self.rag.search(
                search_query,
                top_k=top_k,
                use_reranker=use_reranker,
                title_filter=title_filter
            )
        else:  # bm25
            results = self.rag.search_bm25(
                search_query,
                top_k=top_k,
                use_reranker=use_reranker,
                title_filter=title_filter
            )
        
        return {
            'query': query,
            'search_query': search_query,
            'extracted_company': company_name,
            'title_filter': title_filter,
            'results': results,
            'num_results': len(results)
        }
