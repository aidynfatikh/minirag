#!/usr/bin/env python3
"""
PDF Parser Script for MiniRAG
Parses PDFs from pdfs/ directory to parsed_data/ folder using LLM for title extraction
"""

import sys
import signal
from pathlib import Path
import argparse
import json
from typing import List, Literal, Optional, Tuple, Union

# Add parent directory to path to import minirag
sys.path.insert(0, str(Path(__file__).parent.parent))

from minirag import get_config
from minirag.pdf_parser import parse_pdf, save_parsed_data, _generate_fallback_title_simple
from minirag.generator import Generator

# Global flag for graceful shutdown
_shutdown_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully"""
    global _shutdown_requested
    if _shutdown_requested:
        print("\n\nForce quit!")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\nShutdown requested... cleaning up (press Ctrl-C again to force quit)")

def parse_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    llm_extractor: Optional[Generator] = None,
    use_llm: bool = True,
    force: bool = False,
    verbose: bool = False,
) -> Union[Tuple[Path, str, int], Literal['skipped'], None]:
    """Parse a single PDF file.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save parsed JSON
        llm_extractor: Shared Generator instance for LLM title extraction (used for all PDFs)
        use_llm: Whether to use LLM for title extraction
        force: If False, skip already-parsed PDFs
        verbose: If True, print per-page progress and title extraction details

    Returns:
        (output_path, title, corrupted_pages_count) on success, 'skipped' if skipped, None if failed
    """
    global _shutdown_requested
    
    if _shutdown_requested:
        return None
    
    # Check if already parsed
    output_path = output_dir / f"{pdf_path.stem}.json"
    if output_path.exists() and not force:
        return 'skipped'
    
    try:
        try:
            pages_data, title, corrupted_pages_count = parse_pdf(
                pdf_path,
                extract_title_with_llm=use_llm,
                verbose=verbose,
                generator=llm_extractor,
            )
        except Exception as e:
            # Try simpler extraction method
            try:
                pages_data, title, corrupted_pages_count = parse_pdf(
                    pdf_path,
                    extract_title_with_llm=False,
                    verbose=verbose,
                    simple_mode=True,
                    generator=None,
                )
            except Exception as e2:
                raise Exception(f"Both extraction methods failed: {e2}")

        # When PDF has encoding issues, use safe title from first page only
        if corrupted_pages_count > 0:
            title = _generate_fallback_title_simple(pages_data)
            if not title or title == "Document":
                title = "Unknown (parsing issues)"

        # Update pages with title in hierarchy if needed
        if title and title != "Unknown":
            from minirag.pdf_parser import HierarchyTracker
            hierarchy = HierarchyTracker(title=title)

            for page_data in pages_data:
                # Re-process headers to update hierarchy with title
                for header in page_data['metadata']['headers']:
                    if header['level'] in [1, 2, 3]:
                        hierarchy.update(header['text'], header['level'])

                # Update paths with title included
                page_data['hierarchy_path'] = hierarchy.get_path()
                page_data['hierarchy_context'] = hierarchy.format_context()
                page_data['metadata']['section'] = (
                    hierarchy.get_path()[-1] if hierarchy.get_path() else None
                )

        save_parsed_data(
            pages_data,
            pdf_path.name,
            output_path,
            title,
            corrupted_pages_count=corrupted_pages_count,
        )
        return (output_path, title or "Unknown", corrupted_pages_count)

    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        print(f"  failed: {e}")
        return None

def main():
    """Main function for parsing PDFs"""
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='Parse PDFs to JSON with LLM title extraction')
    parser.add_argument(
        '--pdfs-dir',
        type=str,
        default=None,
        help='Directory containing PDF files (default: config pdfs_dir)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save parsed JSON files (default: config parsed_data_dir)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM for title extraction (use simple header-based extraction)'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Parse only a specific PDF file'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reparse even if JSON already exists (default: skip existing)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print per-page progress and title extraction details'
    )
    args = parser.parse_args()
    
    # Get project root and config
    project_root = Path(__file__).resolve().parent.parent
    config = get_config()
    
    # Set directories
    pdfs_dir = Path(args.pdfs_dir) if args.pdfs_dir else project_root / config.paths.pdfs_dir
    output_dir = Path(args.output_dir) if args.output_dir else project_root / config.paths.parsed_data_dir
    
    # Ensure directories exist
    if not pdfs_dir.exists():
        print(f"❌ PDFs directory not found: {pdfs_dir}")
        print(f"Please create the directory and add PDF files.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get PDF files to parse
    if args.file:
        pdf_file = Path(args.file)
        if not pdf_file.exists():
            # Try relative to pdfs_dir
            pdf_file = pdfs_dir / pdf_file.name
        if not pdf_file.exists():
            print(f"❌ PDF file not found: {args.file}")
            return
        pdf_files = [pdf_file]
    else:
        pdf_files = sorted(pdfs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files in {pdfs_dir}")
        return

    # Load LLM once if needed
    llm_extractor = None
    if not args.no_llm:
        try:
            print("Loading LLM for title extraction...")
            llm_extractor = Generator()
        except Exception as e:
            print(f"LLM init failed: {e}. Using simple title extraction.")

    print(f"Parsing {len(pdf_files)} PDF(s) -> {output_dir} (LLM: {'on' if llm_extractor else 'off'}, force={args.force})\n")

    successful = 0
    failed = 0
    skipped = 0
    parsing_issues_list: List[str] = []

    try:
        for i, pdf_file in enumerate(pdf_files, 1):
            if _shutdown_requested:
                break
            out = parse_single_pdf(
                pdf_file,
                output_dir,
                llm_extractor,
                use_llm=(not args.no_llm),
                force=args.force,
                verbose=args.verbose,
            )
            if out == 'skipped':
                skipped += 1
                print(f"  [{i}/{len(pdf_files)}] {pdf_file.name} (skipped)")
            elif out:
                output_path, title, corrupted_count = out
                successful += 1
                suffix = f"  |  {title}"
                if corrupted_count > 0:
                    suffix += f"  [parsing issues: {corrupted_count} pages unreadable]"
                print(f"  [{i}/{len(pdf_files)}] {pdf_file.name} -> {output_path.name}{suffix}")
                if corrupted_count > 0:
                    parsing_issues_list.append(pdf_file.name)
            else:
                failed += 1
                print(f"  [{i}/{len(pdf_files)}] {pdf_file.name} (failed)")
    finally:
        if llm_extractor:
            llm_extractor.cleanup()

    print(f"\nDone: {successful} parsed, {skipped} skipped, {failed} failed")
    if parsing_issues_list:
        print(f"PDFs with parsing issues ({len(parsing_issues_list)}):")
        for name in parsing_issues_list:
            print(f"  - {name}")

if __name__ == "__main__":
    main()
