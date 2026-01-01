import fitz
from pathlib import Path
import json

def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for page_num, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text:
            pages_data.append({
                'page': page_num,
                'text': text
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
        test_pdf = pdf_files[0]
        print(f"Parsing: {test_pdf.name}")
        
        output_path = process_pdf(test_pdf, output_dir)
        text = output_path.read_text(encoding="utf-8")
        
        print(f"Saved to: {output_path}")
        print(f"Extracted {len(text)} characters")
        print("=" * 80)
        print(text[:500] + "..." if len(text) > 500 else text)
    else:
        print("No PDF files found in pdfs/ directory")
