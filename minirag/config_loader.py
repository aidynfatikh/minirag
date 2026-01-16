"""Configuration loader for MiniRAG"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    device: str = "cuda"
    enabled: bool = True
    quantize: bool = False


@dataclass
class HNSWConfig:
    """HNSW index configuration"""
    m: int = 32
    ef_construction: int = 256
    ef_search: int = 50


@dataclass
class IndexingConfig:
    """Indexing configuration"""
    chunk_size: int = 256
    overlap: int = 32
    use_summary: bool = False
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)


@dataclass
class HybridSearchConfig:
    """Hybrid search configuration"""
    embedding_weight: float = 0.5
    bm25_weight: float = 0.5
    candidates_per_method: int = 50


@dataclass
class SearchConfig:
    """Search configuration"""
    top_k: int = 5
    retrieve_multiplier: int = 4
    company_filter_multiplier: int = 3
    hybrid: HybridSearchConfig = field(default_factory=HybridSearchConfig)
    section_boost_weight: float = 2.0


@dataclass
class GenerationConfig:
    """Generation configuration"""
    max_tokens: int = 800
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    use_history: bool = True
    history_window: int = 6
    max_context_chunks: int = 8
    max_input_length: int = 6144
    
    @dataclass
    class CompanyExtraction:
        max_tokens: int = 30
        temperature: float = 0.1
        max_input_length: int = 2048
        preview_chars: int = 3000
    
    @dataclass
    class TitleExtraction:
        max_tokens: int = 100
        temperature: float = 0.1
        max_input_length: int = 4096
        preview_pages: int = 3
        preview_chars_per_page: int = 2000
        max_pages_to_check: int = 10
        increment_pages: int = 2
    
    company_extraction: CompanyExtraction = field(default_factory=CompanyExtraction)
    title_extraction: TitleExtraction = field(default_factory=TitleExtraction)


@dataclass
class HeaderDetectionConfig:
    """Header detection configuration"""
    font_size_multiplier: float = 1.2
    large_header_multiplier: float = 0.95
    medium_header_multiplier: float = 1.5
    max_header_words: int = 10


@dataclass
class ChunkingConfig:
    """Chunking configuration"""
    min_chunk_ratio: float = 0.5
    max_chunk_ratio: float = 1.5
    max_header_words: int = 15
    fallback_max_chars: int = 1000


@dataclass
class DocumentProcessingConfig:
    """Document processing configuration"""
    header_detection: HeaderDetectionConfig = field(default_factory=HeaderDetectionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


@dataclass
class PathsConfig:
    """File paths configuration"""
    pdfs_dir: str = "pdfs"
    parsed_data_dir: str = "parsed_data"
    index_file: str = "vdb.index"
    logs_dir: str = "logs"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_timestamp_format: str = "%Y%m%d_%H%M%S"


@dataclass
class CompanyDetectionConfig:
    """Company detection configuration"""
    keywords: List[str] = field(default_factory=lambda: [
        "CORPORATION", "COMPANY", "INC.", "INC", "LTD", "GROUP", "CO."
    ])
    min_company_name_length: int = 2
    max_company_name_length: int = 60
    preview_lines: int = 15
    check_headers_count: int = 3
    min_word_length_for_partial_match: int = 3


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    batch_size: int = 32
    show_progress_bar: bool = True
    normalize_embeddings: bool = True


@dataclass
class Config:
    """Main configuration class"""
    models: Dict[str, Any] = field(default_factory=dict)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    company_detection: CompanyDetectionConfig = field(default_factory=CompanyDetectionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    @classmethod
    def from_yaml(cls, path: str = "config/config.yaml") -> "Config":
        """Load configuration from YAML file"""
        config_path = Path(path)
        if not config_path.exists():
            # Try alternate locations
            alt_paths = ["config.yaml", "../config/config.yaml"]
            for alt_path in alt_paths:
                alt_config = Path(alt_path)
                if alt_config.exists():
                    config_path = alt_config
                    break
            else:
                print(f"Warning: Config file not found in {path} or alternate locations, using defaults")
                return cls()
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse nested configurations
        config = cls()
        
        if 'models' in data:
            config.models = data['models']
        
        if 'indexing' in data:
            idx_data = data['indexing']
            hnsw_data = idx_data.get('hnsw', {})
            config.indexing = IndexingConfig(
                chunk_size=idx_data.get('chunk_size', 256),
                overlap=idx_data.get('overlap', 32),
                use_summary=idx_data.get('use_summary', False),
                hnsw=HNSWConfig(
                    m=hnsw_data.get('m', 32),
                    ef_construction=hnsw_data.get('ef_construction', 256),
                    ef_search=hnsw_data.get('ef_search', 50)
                )
            )
        
        if 'search' in data:
            search_data = data['search']
            hybrid_data = search_data.get('hybrid', {})
            config.search = SearchConfig(
                top_k=search_data.get('top_k', 5),
                retrieve_multiplier=search_data.get('retrieve_multiplier', 4),
                company_filter_multiplier=search_data.get('company_filter_multiplier', 3),
                hybrid=HybridSearchConfig(
                    embedding_weight=hybrid_data.get('embedding_weight', 0.5),
                    bm25_weight=hybrid_data.get('bm25_weight', 0.5),
                    candidates_per_method=hybrid_data.get('candidates_per_method', 50)
                ),
                section_boost_weight=search_data.get('section_boost_weight', 2.0)
            )
        
        if 'generation' in data:
            gen_data = data['generation']
            comp_extract = gen_data.get('company_extraction', {})
            title_extract = gen_data.get('title_extraction', {})
            config.generation = GenerationConfig(
                max_tokens=gen_data.get('max_tokens', 800),
                temperature=gen_data.get('temperature', 0.7),
                top_p=gen_data.get('top_p', 0.9),
                repetition_penalty=gen_data.get('repetition_penalty', 1.1),
                use_history=gen_data.get('use_history', True),
                history_window=gen_data.get('history_window', 6),
                max_context_chunks=gen_data.get('max_context_chunks', 8),
                max_input_length=gen_data.get('max_input_length', 6144),
                company_extraction=GenerationConfig.CompanyExtraction(
                    max_tokens=comp_extract.get('max_tokens', 30),
                    temperature=comp_extract.get('temperature', 0.1),
                    max_input_length=comp_extract.get('max_input_length', 2048),
                    preview_chars=comp_extract.get('preview_chars', 3000)
                ),
                title_extraction=GenerationConfig.TitleExtraction(
                    max_tokens=title_extract.get('max_tokens', 50),
                    temperature=title_extract.get('temperature', 0.1),
                    max_input_length=title_extract.get('max_input_length', 2048),
                    preview_pages=title_extract.get('preview_pages', 3),
                    preview_chars_per_page=title_extract.get('preview_chars_per_page', 1000),
                    max_pages_to_check=title_extract.get('max_pages_to_check', 10),
                    increment_pages=title_extract.get('increment_pages', 2)
                )
            )
        
        if 'document_processing' in data:
            doc_data = data['document_processing']
            header_data = doc_data.get('header_detection', {})
            chunk_data = doc_data.get('chunking', {})
            config.document_processing = DocumentProcessingConfig(
                header_detection=HeaderDetectionConfig(
                    font_size_multiplier=header_data.get('font_size_multiplier', 1.2),
                    large_header_multiplier=header_data.get('large_header_multiplier', 0.95),
                    medium_header_multiplier=header_data.get('medium_header_multiplier', 1.5),
                    max_header_words=header_data.get('max_header_words', 10)
                ),
                chunking=ChunkingConfig(
                    min_chunk_ratio=chunk_data.get('min_chunk_ratio', 0.5),
                    max_chunk_ratio=chunk_data.get('max_chunk_ratio', 1.5),
                    max_header_words=chunk_data.get('max_header_words', 15),
                    fallback_max_chars=chunk_data.get('fallback_max_chars', 1000)
                )
            )
        
        if 'paths' in data:
            paths_data = data['paths']
            config.paths = PathsConfig(**paths_data)
        
        if 'logging' in data:
            log_data = data['logging']
            config.logging = LoggingConfig(**log_data)
        
        if 'company_detection' in data:
            comp_data = data['company_detection']
            config.company_detection = CompanyDetectionConfig(**comp_data)
        
        if 'performance' in data:
            perf_data = data['performance']
            config.performance = PerformanceConfig(**perf_data)
        
        return config
    
    def get_model_config(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific model type"""
        return self.models.get(model_type)


def load_test_cases(path: str = "config/test_cases.json") -> List[Dict[str, Any]]:
    """Load test cases from JSON file"""
    test_path = Path(path)
    if not test_path.exists():
        # Try alternate locations
        alt_paths = ["test_cases.json", "../config/test_cases.json"]
        for alt_path in alt_paths:
            alt_test = Path(alt_path)
            if alt_test.exists():
                test_path = alt_test
                break
        else:
            print(f"Warning: Test cases file not found in {path} or alternate locations")
            return []
    
    with open(test_path, 'r') as f:
        return json.load(f)


# Global configuration instance
_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None or reload:
        _config = Config.from_yaml()
    return _config


def reload_config():
    """Reload configuration from file"""
    return get_config(reload=True)


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Embedding model: {config.models.get('embedding', {}).get('name')}")
    print(f"Chunk size: {config.indexing.chunk_size}")
    print(f"HNSW M: {config.indexing.hnsw.m}")
    print(f"Search top_k: {config.search.top_k}")
    print(f"Generation max_tokens: {config.generation.max_tokens}")
    
    # Test test cases loading
    test_cases = load_test_cases()
    print(f"\nLoaded {len(test_cases)} test cases")
    if test_cases:
        print(f"First test case: {test_cases[0]['query']}")
