"""
MiniRAG - Lightweight Retrieval-Augmented Generation System

A production-ready RAG system with:
- Hybrid search (dense + sparse)
- Cross-encoder reranking
- Section-aware retrieval
- Conversational generation
"""

from .rag import RAG
from .generator import Generator
from .pdf_parser import parse_pdf, process_pdf, HierarchyTracker
from .config_loader import (
    get_config,
    load_test_cases,
    Config,
    ModelConfig,
    IndexingConfig,
    SearchConfig,
    GenerationConfig,
)

__version__ = "1.0.0"
__all__ = [
    "RAG",
    "Generator",
    "parse_pdf",
    "process_pdf",
    "HierarchyTracker",
    "get_config",
    "load_test_cases",
    "Config",
]
