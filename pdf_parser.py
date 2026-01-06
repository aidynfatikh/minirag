import fitz
from pathlib import Path
import json
import re
from collections import defaultdict

class HierarchyTracker:
    """Tracks document hierarchy as we parse"""
    def __init__(self):
        self.hierarchy = {}  # level -> current header
        self.company = None
        
    def update(self, header_text, level):
        """Update hierarchy when we see a new header"""
        # Clear deeper levels when we see a higher level header
        self.hierarchy = {k: v for k, v in self.hierarchy.items() if k < level}
        self.hierarchy[level] = header_text
        
        # Detect company name from first page headers
        if not self.company and level == 1 and any(word in header_text.upper() for word in ['CORPORATION', 'COMPANY', 'GROUP', 'INC']):
            self.company = header_text
    
    def get_path(self):
        """Get current hierarchical path"""
        path = []
        
        if self.company:
            path.append(f"Company: {self.company}")
        
        # Add hierarchy in order of levels
        for level in sorted(self.hierarchy.keys()):
            header = self.hierarchy[level]
            if level == 1:
                # Check if it's a Part/Section indicator
                if re.match(r'^(Part|Section|Chapter|Article)\s+\d+', header, re.IGNORECASE):
                    path.append(f"Part: {header}")
                elif header != self.company:  # Don't duplicate company
                    path.append(f"Section: {header}")
            elif level == 2:
                path.append(f"Section: {header}")
            elif level == 3:
                path.append(f"Subsection: {header}")
        
        return path
    
    def format_context(self):
        """Format hierarchy as context string"""
        path = self.get_path()
        if not path:
            return ""
        return '\n'.join(path) + '\n\n'

def detect_headers_and_sections(page, page_num):
    """
    Detect headers and sections based on font properties and text patterns.
    Returns a list of text blocks with metadata including section/header info.
    """
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
        if font_size > avg_font_size * 1.2:
            is_header = True
            if font_size >= max_font_size * 0.95:
                header_level = 1
            elif font_size >= avg_font_size * 1.5:
                header_level = 2
            else:
                header_level = 3
        
        # Check for bold text at start of line
        elif is_bold and len(text.split()) <= 10:
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

def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_data = []
    hierarchy = HierarchyTracker()
    
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
                    hierarchy.update(block['text'], block['header_level'])
        
        # Get current hierarchical context
        hierarchy_path = hierarchy.get_path()
        hierarchy_context = hierarchy.format_context()
        
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
    return pages_data

def save_parsed_data(pages_data, pdf_name, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'pdf_name': pdf_name,
        'total_pages': len(pages_data),
        'pages': pages_data
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def process_pdf(pdf_path, output_dir):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    pages_data = parse_pdf(pdf_path)
    output_path = output_dir / f"{pdf_path.stem}.json"
    save_parsed_data(pages_data, pdf_path.name, output_path)
    
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
