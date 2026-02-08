import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional
import os
from pathlib import Path

from .config_loader import get_config, Config, GenerationConfig

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed


class Generator:
    """Text generation using LLM for RAG responses"""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        quantize: Optional[bool] = None
    ):
        """Initialize generator
        
        Args:
            config: Configuration object (if None, loads from default config file)
            model_name: Override model name
            device: Override device
            quantize: Override quantization setting
        """
        self.config = config or get_config()
        self.gen_config = self.config.generation
        
        # Get model config
        gen_model_config = self.config.models.get('generator', {})
        
        # Use provided values or fall back to config
        model_name = model_name or gen_model_config.get('name', 'meta-llama/Llama-3.2-3B-Instruct')
        self.device = device or gen_model_config.get('device', 'cuda')
        quantize = quantize if quantize is not None else gen_model_config.get('quantize', True)
        
        self.conversation_history = []
        
        # Get HuggingFace token if available
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        
        # Try to use local files first, fall back to download if not available
        try:
            print(f"Loading {model_name} (quantized: {quantize})...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                local_files_only=True,
                token=hf_token
            )
        except Exception as e:
            print(f"  Model not cached locally, downloading from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if quantize and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    token=hf_token,
                )
            except Exception as e:
                print(f"  Model not cached locally, downloading from HuggingFace...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    token=hf_token,
                )
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map=self.device,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    token=hf_token,
                )
            except Exception as e:
                print(f"  Model not cached locally, downloading from HuggingFace...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map=self.device,
                    low_cpu_mem_usage=True,
                    token=hf_token,
                )
        self.model.eval()
        print(f"Model loaded")
    
    def cleanup(self):
        """Clean up model and free GPU memory"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Model cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def __del__(self):
        self.cleanup()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def generate(
        self,
        query: str,
        chunks: List[Dict],
        max_tokens: Optional[int] = None,
        temp: Optional[float] = None,
        use_history: Optional[bool] = None
    ) -> Dict:
        """Generate answer from query and retrieved chunks
        
        Args:
            query: User query
            chunks: Retrieved context chunks
            max_tokens: Maximum tokens to generate (uses config default if None)
            temp: Temperature for generation (uses config default if None)
            use_history: Whether to use conversation history (uses config default if None)
        
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        if not chunks:
            return {"answer": "No context available", "sources": []}
        
        # Use config defaults
        max_tokens = max_tokens or self.gen_config.max_tokens
        temp = temp if temp is not None else self.gen_config.temperature
        use_history = use_history if use_history is not None else self.gen_config.use_history
        max_context_chunks = self.gen_config.max_context_chunks
        
        # Format context from retrieved chunks
        context = "\n\n".join([
            f"[Document {i+1}]\nSource: {c['pdf_name']}, Page {c['page']}\nContent: {c.get('chunk', c.get('text', ''))}"
            for i, c in enumerate(chunks[:max_context_chunks])
        ])
        
        # Build conversation with system prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant with access to a document database. "
                    "Answer questions accurately based on the provided context. "
                    "If you need more specific information, you can ask follow-up questions. "
                    "Be thorough but concise. Cite page numbers when referencing specific information."
                )
            }
        ]
        
        # Add conversation history if enabled
        if use_history and self.conversation_history:
            history_window = self.gen_config.history_window
            messages.extend(self.conversation_history[-history_window:])  # Keep last N exchanges
        
        # Add current context and query
        messages.append({
            "role": "user",
            "content": f"Context from documents:\n{context}\n\nQuestion: {query}"
        })
        
        # Format using chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        max_input_length = self.gen_config.max_input_length
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
                top_p=self.gen_config.top_p,
                repetition_penalty=self.gen_config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Update conversation history
        if use_history:
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
        
        sources = []
        seen = set()
        for c in chunks:
            key = (c['pdf_name'], c['page'])
            if key not in seen:
                seen.add(key)
                sources.append({"pdf": c['pdf_name'], "page": c['page']})
        
        return {"answer": answer, "sources": sources}
    
    def extract_title(
        self,
        pages_data: List[Dict],
        max_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """Extract document title from pages using LLM with incremental page checking.

        If initial pages don't yield a valid title, incrementally checks more pages
        until a title is found or max pages reached.

        Args:
            pages_data: List of page dictionaries with 'page', 'text', and 'metadata'
            max_tokens: Maximum tokens for response (uses config default if None)
            verbose: If True, print extraction attempts and result

        Returns:
            Extracted title (never returns 'Unknown')
        """
        title_extract_config = self.gen_config.title_extraction
        max_tokens = max_tokens or title_extract_config.max_tokens
        preview_pages = title_extract_config.preview_pages
        preview_chars_per_page = title_extract_config.preview_chars_per_page
        max_pages = min(title_extract_config.max_pages_to_check, len(pages_data))
        increment = title_extract_config.increment_pages
        
        current_pages = preview_pages
        attempt = 1
        
        while current_pages <= max_pages:
            if verbose:
                print(f"  Attempting title extraction with {current_pages} pages (attempt {attempt})...")
            
            # Collect FULL text from current number of pages - don't hide anything
            preview_text_parts = []
            for page_data in pages_data[:current_pages]:
                page_num = page_data.get('page', 0)
                # Get full text up to limit, don't truncate important content
                text = page_data.get('text', '')[:preview_chars_per_page]
                
                # Include ALL headers for better context
                headers = page_data.get('metadata', {}).get('headers', [])
                if headers:
                    header_texts = [h['text'] for h in headers]
                    preview_text_parts.append(f"=== PAGE {page_num} ===\nHEADERS: {' | '.join(header_texts)}\n\nCONTENT:\n{text}")
                else:
                    preview_text_parts.append(f"=== PAGE {page_num} ===\nCONTENT:\n{text}")
            
            preview_text = '\n\n'.join(preview_text_parts)

            # If extracted text looks corrupted (PDF encoding/font issue), skip LLM and use fallback
            if self._preview_text_looks_corrupted(preview_text):
                if verbose:
                    print(f"  Extracted text looks corrupted (encoding/font), using fallback title.")
                break

            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at extracting DISTINCTIVE document titles. "
                        "Your goal is to create a UNIQUE title that distinguishes this document from others. "
                        "\n\nRULES:"
                        "\n1. ALWAYS include the ORGANIZATION/ENTITY NAME if visible"
                        "\n2. Include the document TYPE (Annual Report, Form 10-K, Financial Statements, etc.)"
                        "\n3. Include the YEAR if present"
                        "\n4. Format: '[Entity Name] [Document Type] [Year]' or similar distinctive format"
                        "\n\nEXAMPLES of GOOD titles:"
                        "\n- 'Apple Inc. Annual Report 2022'"
                        "\n- 'Microsoft Corporation Form 10-K 2023'"
                        "\n- 'Tesla Inc. Financial Statements Q4 2022'"
                        "\n- 'Amazon.com Inc. Proxy Statement 2023'"
                        "\n\nEXAMPLES of BAD titles (too generic):"
                        "\n- 'Annual Report' (missing entity name)"
                        "\n- '2022 Annual Report' (missing entity name)"
                        "\n- 'Form 10-K' (missing entity name)"
                        "\n\nReturn ONLY the title, nothing else. If you truly cannot find any identifying information, return 'UNKNOWN_TITLE'."
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract a DISTINCTIVE title (including entity/organization name) from this document:\n\n{preview_text}\n\nDistinctive Title:"
                }
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            max_input_length = title_extract_config.max_input_length
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=title_extract_config.temperature,
                    do_sample=True,
                    top_p=self.gen_config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            title = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Clean up the response
            title = title.strip('"\'.,')
            
            # If response is too long or contains multiple lines, extract first line
            if len(title) > 150 or '\n' in title:
                lines = title.split('\n')
                title = lines[0].strip()
            
            # Remove common prefixes/suffixes in LLM responses
            for phrase in ['The title is ', 'Title: ', 'Document title: ', 'Answer: ', 'Based on']:
                if title.lower().startswith(phrase.lower()):
                    title = title[len(phrase):].strip()
                    break
            
            # Check if we got a valid, plausible title (reject garbage/encoding artifacts)
            invalid_titles = ['none', 'n/a', 'not found', 'unclear', 'unknown', '', 'unknown_title']
            if (
                title
                and title.lower() not in invalid_titles
                and len(title) > 2
                and self._is_plausible_title(title)
            ):
                # Limit to reasonable length
                if len(title) > 150:
                    title = title[:147] + '...'
                if verbose:
                    print(f"  Title: {title}")
                return title

            # If we didn't get a valid title and have more pages to check
            if current_pages < max_pages:
                if verbose:
                    print(f"  No valid title, trying with more pages...")
                current_pages += increment
                attempt += 1
            else:
                break

        # If we exhausted all attempts, use fallback based on first page content
        if verbose:
            print(f"  Using fallback title (LLM did not return valid title).")
        fallback_title = self._generate_fallback_title(pages_data)
        return fallback_title
    
    def _preview_text_looks_corrupted(self, preview_text: str, sample_len: int = 500) -> bool:
        """True if text we extracted from the PDF looks like garbage (wrong encoding/font)."""
        sample = (preview_text or "")[:sample_len]
        if len(sample) < 20:
            return False
        allowed = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\r\t.,'-&();:!/"
        )
        allowed_count = sum(1 for c in sample if c in allowed)
        return allowed_count < 0.5 * len(sample)

    def _is_plausible_title(self, title: str) -> bool:
        """Return False if title looks like garbage (wrong decoding, symbols, no words)."""
        if not title or len(title) < 4:
            return False
        # Reject encoding-garbage symbols (font/ToUnicode corruption)
        garbage_chars = set("=@<>;:\\")
        if sum(1 for c in title if c in garbage_chars) > 2:
            return False
        # Document titles are usually multiple words (letters, digits, spaces, common punctuation)
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,'-&()")
        allowed_count = sum(1 for c in title if c in allowed)
        if allowed_count < 0.7 * len(title):
            return False
        # Should contain at least one space (e.g. "Company Name Annual Report 2022")
        if ' ' not in title:
            return False
        # At least two "words" (letters/digits)
        words = [w for w in title.split() if any(c.isalnum() for c in w)]
        if len(words) < 2:
            return False
        return True

    def _generate_fallback_title(self, pages_data: List[Dict]) -> str:
        """Generate a fallback title from document content.

        Args:
            pages_data: List of page dictionaries

        Returns:
            Fallback title based on document structure
        """
        if not pages_data:
            return "Untitled Document"
        first_meta = pages_data[0].get('metadata') or {}
        # Prefer PDF document metadata title when page extraction was corrupted
        pdf_title = first_meta.get('pdf_metadata_title') or ""
        if pdf_title and self._is_plausible_title(pdf_title):
            return pdf_title[:150]
        # Try to get title from first page headers
        if first_meta.get('headers'):
            first_header = first_meta['headers'][0]
            header_text = first_header.get('text', '').strip()
            if header_text and len(header_text) > 3 and self._is_plausible_title(header_text):
                return header_text[:150]
        # Try to extract from first line of first page (only if it looks readable)
        if pages_data[0].get('text'):
            first_text = pages_data[0]['text'].strip()
            first_line = first_text.split('\n')[0].strip()
            if (
                first_line
                and len(first_line) > 3
                and len(first_line) < 150
                and self._is_plausible_title(first_line)
            ):
                return first_line
        
        # Last resort: use filename-based title
        return "Untitled Document"
