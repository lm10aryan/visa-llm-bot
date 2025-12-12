"""
Model Manager - Flexible LLM interface supporting multiple models
Supports: TinyLlama, Phi-3, OpenAI, Context-Only fallback
"""

import os
from pathlib import Path
from typing import Dict, List

import torch
import yaml

class ModelManager:
    def __init__(self, config_path="config.yaml"):
        """Initialize model manager with config"""
        self.config_path = Path(config_path)
        self.load_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.loaded_models: Dict[str, tuple] = {}
        self.model_type = self.config['model']['type']
        
        print(f"🤖 Model Manager initialized: {self.model_type}")
        self._activate_model(self.model_type)
    
    def load_config(self):
        """Load configuration from YAML"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _activate_model(self, model_type: str):
        """Ensure the requested model is loaded and active"""
        if model_type == "context-only":
            self.model_type = "context-only"
            self.model = None
            self.tokenizer = None
            return
        
        if model_type in self.loaded_models:
            self.model_type = model_type
            self.tokenizer, self.model = self.loaded_models[model_type]
            print(f"   🔁 Using cached {model_type} model")
            return
        
        try:
            tokenizer, model = self._load_model(model_type)
            self.loaded_models[model_type] = (tokenizer, model)
            self.tokenizer, self.model = tokenizer, model
            self.model_type = model_type
        except Exception as e:
            print(f"❌ Failed to load {model_type}: {e}")
            print("   Falling back to context-only mode")
            self.model_type = "context-only"
            self.model = None
            self.tokenizer = None
    
    def _load_model(self, model_type: str):
        """Load the specified model and return tokenizer/model pair"""
        if model_type == "tinyllama":
            return self._load_tinyllama()
        if model_type == "phi3":
            return self._load_phi3()
        if model_type == "openai":
            return self._load_openai()
        
        print(f"⚠️  Unknown model type: {model_type}, using context-only")
        return None, None
    
    def switch_model(self, new_type: str):
        """Switch to a different model at runtime"""
        if new_type == self.model_type:
            return
        
        print(f"🔁 Switching model to {new_type}")
        self._activate_model(new_type)
    
    def _load_tinyllama(self):
        """Load TinyLlama model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = self.config['model']['tinyllama']['model_name']
        print(f"   Loading TinyLlama from {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        model = model.to(self.device)
        print("   ✅ TinyLlama loaded successfully")
        return tokenizer, model
    
    def _load_phi3(self):
        """Load Phi-3 model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = self.config['model']['phi3']['model_name']
        print(f"   Loading Phi-3 from {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        model = model.to(self.device)
        print("   ✅ Phi-3 loaded successfully")
        return tokenizer, model
    
    def _load_openai(self):
        """Setup OpenAI API"""
        try:
            import openai
            api_key = self.config['model']['openai'].get('api_key') or os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                raise ValueError("OpenAI API key not found in config or environment")
            
            openai.api_key = api_key
            print("   ✅ OpenAI API configured")
            return None, "openai"
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def generate_response(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """
        Generate response based on query and context
        
        Args:
            query: User's question
            context: Retrieved RAG context
            conversation_history: Previous Q&A pairs
            
        Returns:
            Generated response
        """
        if self.model_type == "context-only":
            structured = self._format_context_only(query, context)
            return self._render_structured_response(structured)
        if self.model_type == "openai":
            return self._generate_openai(query, context, conversation_history)
        return self._generate_local(query, context, conversation_history)
    
    def _render_structured_response(self, structured: Dict) -> str:
        """Render dict output into markdown text"""
        quick = structured.get("quick_answer")
        details = structured.get("details")
        caveats = structured.get("caveats")
        
        parts = []
        if quick:
            parts.append(f"**Quick Answer:** {quick}")
        if details:
            parts.append(str(details).strip())
        if caveats:
            parts.append(f"**Important Notes:** {caveats}")
        
        return "\n\n".join(parts).strip()
    
    def _format_context_only(self, query: str, context: str) -> Dict:
        """Format context without LLM generation"""
        caveat = "Answer grounded strictly in retrieved USCIS text."
        return {
            "quick_answer": "Based on official USCIS sources:",
            "details": context[:800] or "No matching passages retrieved. Please try another query.",
            "caveats": caveat
        }
    
    def _generate_openai(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Generate response using OpenAI API"""
        import openai
        
        config = self.config['model']['openai']
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        
        if conversation_history:
            for item in conversation_history[-3:]:
                messages.append({"role": "user", "content": item.get('user', '')})
                messages.append({"role": "assistant", "content": item.get('assistant', '')})
        
        user_message = f"""Based on this official information:

{context}

User Question: {query}

Provide a helpful, accurate response with:
1. A brief 2-3 sentence answer
2. Key details in bullet points
3. Important caveats or conditions
4. DO NOT make up information not in the context"""

        messages.append({"role": "user", "content": user_message})
        
        try:
            response = openai.ChatCompletion.create(
                model=config['model_name'],
                messages=messages,
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return self._render_structured_response(self._format_context_only(query, context))
    
    def _generate_local(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """Generate response with local TinyLlama/Phi-3 models"""
        config = self.config['model'][self.model_type]
        prompt = f"""You are a precise USCIS immigration assistant.

Use ONLY the official context to craft a thorough response with:
1. A bold quick answer (2-3 sentences).
2. Key requirements or steps in at least 4 detailed bullet points.
3. Important warnings/edge cases in a final paragraph.
Keep the tone factual, cite the context implicitly, and do not fabricate details.

Context:
{context[:900]}

Question: {query}

Assistant:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1000,
            truncation=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.get("max_tokens", 250),
                temperature=min(config.get("temperature", 0.6), 0.8),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            response = response.strip()
            if not response.endswith("."):
                response += "."
            return response
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            fallback = self._format_context_only(query, context)
            return self._render_structured_response(fallback)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the assistant"""
        return """You are a helpful US immigration assistant. 

Your role:
- Answer questions about US visas based ONLY on provided official information
- Be clear, accurate, and helpful
- Structure responses with: Quick Answer, Key Details, Important Notes
- Always mention sources are from official USCIS
- Never make up information not in the context
- If unsure, say so and direct to official sources

Remember: This is informational only, not legal advice."""

    def get_model_info(self) -> Dict:
        """Get information about current model"""
        return {
            "type": self.model_type,
            "loaded": self.model is not None,
            "device": self.device,
            "config": self.config['model'].get(self.model_type, {})
        }


# Test function
if __name__ == "__main__":
    print("Testing Model Manager...")
    
    manager = ModelManager()
    print(f"\n📊 Model Info: {manager.get_model_info()}")
    
    test_query = "What is F-1 OPT?"
    test_context = "OPT (Optional Practical Training) allows F-1 students to work for 12 months after graduation."
    
    print(f"\n🧪 Test Query: {test_query}")
    print("⏳ Generating response...")
    
    response = manager.generate_response(test_query, test_context)
    print(f"\n✅ Response:\n{response}")
