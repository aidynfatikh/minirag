import fitz
from pathlib import Path
import json
import re
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

from .config_loader import get_config

class HierarchyTracker:
    """Tracks document hierarchy as we parse"""
    
    def __init__(self, title: Optional[str] = None):
        self.hierarchy: Dict[int, str] = {}  # level -> current header
        self.title: Optional[str] = title
        self.config = get_config()
        
    def update(self, header_text: str, level: int) -> None:
        """Update hierarchy when we see a new header
        
        Args:
            header_text: Text of the header
            level: Header level (1, 2, 3, etc.)
        """
        # Clear deeper levels when we see a higher level header
        self.hierarchy = {k: v for k, v in self.hierarchy.items() if k < level}
        self.hierarchy[level] = header_text
    
    def get_path(self) -> List[str]:
        """Get current hierarchical path
        
        Returns:
            List of hierarchy path elements
        """
        path = []
        
        if self.title:
            path.append(f"Title: {self.title}")
        
        # Add hierarchy in order of levels
        for level in sorted(self.hierarchy.keys()):
            header = self.hierarchy[level]
            if level == 1:
                # Check if it's a Part/Section indicator
                if re.match(r'^(Part|Section|Chapter|Article)\s+\d+', header, re.IGNORECASE):
                    path.append(f"Part: {header}")
                elif header != self.title:  # Don't duplicate title
                    path.append(f"Section: {header}")
            elif level == 2:
                path.append(f"Section: {header}")
            elif level == 3:
                path.append(f"Subsection: {header}")
        
        return path
    
    def format_context(self) -> str:
        """Format hierarchy as context string
        
        Returns:
            Formatted hierarchy string
        """
        path = self.get_path()
        if not path:
            return ""
        return '\n'.join(path) + '\n\n'

def detect_headers_and_sections(page, page_num: int, config=None) -> List[Dict]:
    """
    Detect headers and sections based on font properties and text patterns.
    
    Args:
        page: PyMuPDF page object
        page_num: Page number
        config: Configuration object (optional)
    
    Returns:
        List of text blocks with metadata including section/header info
    """
    if config is None:
        config = get_config()
    
    header_config = config.document_processing.header_detection
    blocks = page.get_text("dict")["blocks"]
    
    # Collect font information
    font_sizes = []
    text_blocks = []
    
    for block in blocks:
        if block.get("type") == 0:  # Text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        font_size = span.get("size", 0)
                        font_flags = span.get("flags", 0)
                        font_name = span.get("font", "")
                        
                        font_sizes.append(font_size)
                        text_blocks.append({
                            'text': text,
                            'font_size': font_size,
                            'font_flags': font_flags,
                            'font_name': font_name,
                            'bbox': span.get("bbox", []),
                            'is_bold': bool(font_flags & 2**4),  # Bold flag
                            'is_italic': bool(font_flags & 2**1),  # Italic flag
                        })
    
    # Calculate statistics
    if not font_sizes:
        return []
    
    avg_font_size = sum(font_sizes) / len(font_sizes)
    max_font_size = max(font_sizes)
    
    # Classify text blocks
    classified_blocks = []
    current_section = None
    
    for block in text_blocks:
        text = block['text']
        font_size = block['font_size']
        is_bold = block['is_bold']
        
        # Detect if this is a header
        is_header = False
        header_level = None
        
        # Check font size (headers are usually larger)
        font_threshold = avg_font_size * header_config.font_size_multiplier
        if font_size > font_threshold:
            is_header = True
            large_threshold = max_font_size * header_config.large_header_multiplier
            medium_threshold = avg_font_size * header_config.medium_header_multiplier
            
            if font_size >= large_threshold:
                header_level = 1
            elif font_size >= medium_threshold:
                header_level = 2
            else:
                header_level = 3
        
        # Check for bold text at start of line
        elif is_bold and len(text.split()) <= header_config.max_header_words:
            is_header = True
            header_level = 3
        
        # Check for numbered sections (e.g., "1.", "1.1", "Chapter 1")
        section_pattern = re.match(r'^(\d+\.?)+\s+[A-Z]', text)
        chapter_pattern = re.match(r'^(Chapter|Section|Part|Article)\s+\d+', text, re.IGNORECASE)
        
        if section_pattern or chapter_pattern:
            is_header = True
            header_level = 2 if section_pattern else 1
        
        # Update current section
        if is_header:
            current_section = text
        
        classified_blocks.append({
            'text': text,
            'font_size': font_size,
            'is_bold': is_bold,
            'is_italic': block['is_italic'],
            'is_header': is_header,
            'header_level': header_level,
            'section': current_section,
            'page': page_num
        })
    
    return classified_blocks

def _generate_fallback_title_simple(pages_data: List[Dict]) -> str:
    """Generate a fallback title from document content without LLM
    
    Args:
        pages_data: List of page dictionaries
        
    Returns:
        Fallback title based on document structure
    """
    # Try to get title from first page headers
    if pages_data and pages_data[0].get('metadata', {}).get('headers'):
        headers = pages_data[0]['metadata']['headers']
        if headers:
            first_header = headers[0]
            header_text = first_header.get('text', '').strip()
            if header_text and len(header_text) > 3:
                return header_text[:150]
    
    # Try to extract from first line of first page
    if pages_data and pages_data[0].get('text'):
        first_text = pages_data[0]['text'].strip()
        first_line = first_text.split('\n')[0].strip()
        if first_line and len(first_line) > 3 and len(first_line) < 150:
            return first_line
    
    # Check headers from first few pages
    for page_data in pages_data[:5]:
        headers = page_data.get('metadata', {}).get('headers', [])
        for header in headers:
            header_text = header.get('text', '').strip()
            if header_text and len(header_text) > 3 and len(header_text) < 150:
                return header_text
    
    # Last resort: use generic title
    return "Document"

def parse_pdf(pdf_path: Path, extract_title_with_llm: bool = False) -> Tuple[List[Dict], Optional[str]]:
    """Parse PDF and extract text with hierarchy
    
    Args:
        pdf_path: Path to PDF file
        extract_title_with_llm: Whether to use LLM for title extraction (requires Generator)
    
    Returns:
        Tuple of (pages_data list, title)
    """
    doc = fitz.open(pdf_path)
    pages_data = []
    
    # First pass: extract all pages without title
    temp_hierarchy = HierarchyTracker(title=None)
    
    for page_num, page in enumerate(doc, 1):
        # Use "text" layout option for better text extraction with proper spacing
        text = page.get_text("text").strip()
        
        # Fallback to dict method if text is empty
        if not text:
            text = page.get_text().strip()
        
        # Detect headers and sections
        blocks_with_metadata = detect_headers_and_sections(page, page_num)
        
        # Extract headers for this page and update hierarchy
        headers = []
        for block in blocks_with_metadata:
            if block.get('is_header'):
                headers.append({
                    'text': block['text'],
                    'level': block['header_level'],
                    'font_size': block['font_size']
                })
                # Update hierarchy tracker with most significant headers
                if block['header_level'] in [1, 2, 3]:
                    temp_hierarchy.update(block['text'], block['header_level'])
        
        # Get current hierarchical context (without title yet)
        hierarchy_path = temp_hierarchy.get_path()
        hierarchy_context = temp_hierarchy.format_context()
        
        # Always add page, even if text is empty (might be image-only page)
        pages_data.append({
            'page': page_num,
            'text': text if text else f"[Page {page_num} - No extractable text]",
            'hierarchy_path': hierarchy_path,
            'hierarchy_context': hierarchy_context,
            'metadata': {
                'headers': headers,
                'section': hierarchy_path[-1] if hierarchy_path else None,
                'num_headers': len(headers),
                'has_bold': any(b.get('is_bold', False) for b in blocks_with_metadata),
                'has_italic': any(b.get('is_italic', False) for b in blocks_with_metadata),
                'has_text': bool(text)
            }
        })
    
    doc.close()
    
    # Extract title with LLM if requested
    title = None
    if extract_title_with_llm:
        try:
            from .generator import Generator
            print("Extracting title using LLM...")
            with Generator() as generator:
                title = generator.extract_title(pages_data)
            print(f"Extracted title: {title}")
        except Exception as e:
            print(f"Failed to extract title with LLM: {e}")
            import traceback
            traceback.print_exc()
            # Use fallback title instead of Unknown
            title = _generate_fallback_title_simple(pages_data)
            print(f"Using fallback title: {title}")
    else:
        # When LLM is not used, generate title from headers/content
        title = _generate_fallback_title_simple(pages_data)
    
    # Second pass: update hierarchy paths with title
    if title and title != "Unknown":
        hierarchy = HierarchyTracker(title=title)
        
        for page_data in pages_data:
            # Re-process headers to update hierarchy with title
            for header in page_data['metadata']['headers']:
                if header['level'] in [1, 2, 3]:
                    hierarchy.update(header['text'], header['level'])
            
            # Update paths with title included
            page_data['hierarchy_path'] = hierarchy.get_path()
            page_data['hierarchy_context'] = hierarchy.format_context()
            page_data['metadata']['section'] = hierarchy.get_path()[-1] if hierarchy.get_path() else None
    
    return pages_data, title

def save_parsed_data(pages_data: List[Dict], pdf_name: str, output_path: Path, title: Optional[str] = None) -> None:
    """Save parsed data to JSON file
    
    Args:
        pages_data: List of parsed page dictionaries
        pdf_name: Name of the PDF file
        output_path: Path to save JSON output
        title: Document title (optional)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'pdf_name': pdf_name,
        'title': title or 'Unknown',
        'total_pages': len(pages_data),
        'pages': pages_data
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def process_pdf(pdf_path: Path, output_dir: Path, use_llm_for_title: bool = False) -> Path:
    """Process a single PDF file
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save output
        use_llm_for_title: Whether to use LLM for title extraction
    
    Returns:
        Path to output JSON file
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    pages_data, title = parse_pdf(pdf_path, extract_title_with_llm=use_llm_for_title)
    output_path = output_dir / f"{pdf_path.stem}.json"
    save_parsed_data(pages_data, pdf_path.name, output_path, title)
    
    return output_path


if __name__ == "__main__":
    pdfs_dir = Path("pdfs")
    output_dir = Path("parsed_data")
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    
    if pdf_files:
        for file in pdf_files:
            print(f"Parsing: {file.name}")
            
            output_path = process_pdf(file, output_dir)
            text = output_path.read_text(encoding="utf-8")
    else:
        print("No PDF files found in pdfs/ directory")
    
    print("Done parsing")
