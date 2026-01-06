import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional


class Generator:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", device: str = "cuda", quantize: bool = True):
        self.device = device
        self.conversation_history = []
        print(f"Loading {model_name} (quantized: {quantize})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if quantize and device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map=device,
                low_cpu_mem_usage=True,
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
    
    def generate(self, query: str, chunks: List[Dict], max_tokens: int = 800, temp: float = 0.7, use_history: bool = True) -> Dict:
        if not chunks:
            return {"answer": "No context available", "sources": []}
        
        # Format context from retrieved chunks
        context = "\n\n".join([
            f"[Document {i+1}]\nSource: {c['pdf_name']}, Page {c['page']}\nContent: {c.get('chunk', c.get('text', ''))}"
            for i, c in enumerate(chunks[:8])
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
            messages.extend(self.conversation_history[-6:])  # Keep last 3 exchanges
        
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
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6144).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
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
    
    def extract_company_name(self, text: str, max_tokens: int = 30) -> str:
        """Extract company name from document text using LLM
        
        Args:
            text: First 2 pages of document text
            max_tokens: Maximum tokens for response
            
        Returns:
            Extracted company name or 'Unknown'
        """
        prompt_messages = [
            {
                "role": "system",
                "content": "Extract the company name from the document. Return ONLY the company name, nothing else."
            },
            {
                "role": "user",
                "content": f"Document:\n{text[:3000]}\n\nCompany name:"
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,  # Low temperature for more deterministic output
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        company_name = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Clean up the response
        company_name = company_name.strip('"\'.,')
        
        # If response is too long or contains explanation, try to extract just the name
        if len(company_name) > 100 or '\n' in company_name:
            lines = company_name.split('\n')
            company_name = lines[0].strip()
        
        # Remove common prefixes/suffixes in responses
        for phrase in ['The company is ', 'Company name: ', 'Company: ', 'Answer: ']:
            if company_name.startswith(phrase):
                company_name = company_name[len(phrase):].strip()
        
        # Validate result
        if not company_name or company_name.lower() in ['none', 'n/a', 'not found', 'unclear', '']:
            return 'Unknown'
        
        return company_name
