"""
LLM Utilities for PathoDetect+
Enhanced version with LangChain support
"""

import os
import json
from typing import Dict, List, Any, Optional
import requests

try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_community.callbacks.manager import get_openai_callback
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Install with: pip install langchain-community openai")

class LLMProcessor:
    def __init__(self, config):
        self.config = config
        self.langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.model_name = config['llm']['model_name']
        self.temperature = config['llm']['temperature']
        self.max_tokens = config['llm']['max_tokens']
        self.system_prompt = config['llm']['system_prompt']
        
        # Initialize LangChain if available
        self.langchain_model = None
        if LANGCHAIN_AVAILABLE and self.langchain_api_key:
            try:
                # Use LangChain with OpenAI-compatible endpoint
                self.langchain_model = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=self.langchain_api_key,
                    base_url="https://api.groq.com/openai/v1"  # Groq endpoint
                )
                print("✅ LangChain initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize LangChain: {e}")
                self.langchain_model = None
        
        if not self.langchain_api_key and not self.groq_api_key:
            print("Warning: No LLM API keys found. Set LANGCHAIN_API_KEY or GROQ_API_KEY")
    
    def generate_report(self, predictions: List[Dict], image_info: Dict) -> str:
        """Generate a comprehensive pathology report"""
        if not self.langchain_model and not self.groq_api_key:
            return self._generate_fallback_report(predictions, image_info)
        
        try:
            # Prepare context
            cancer_patches = sum(1 for p in predictions if p.get('cancer_prob', 0) > 0.5)
            total_patches = len(predictions)
            avg_confidence = sum(p.get('cancer_prob', 0) for p in predictions) / total_patches if total_patches > 0 else 0
            cancer_rate = (cancer_patches/total_patches*100) if total_patches > 0 else 0
            
            prompt = f"""
            Generate a detailed pathology report based on the following findings:
            
            - Total patches analyzed: {total_patches}
            - Cancerous patches detected: {cancer_patches}
            - Average confidence: {avg_confidence:.3f}
            - Cancer detection rate: {cancer_rate:.1f}%
            
            Please provide:
            1. Summary of findings
            2. Clinical interpretation
            3. Recommendations for further analysis
            4. Confidence assessment
            """
            
            # Try LangChain first, then fallback to Groq
            if self.langchain_model:
                response = self._call_langchain_api(prompt)
            else:
                response = self._call_groq_api(prompt)
                
            return response if response else self._generate_fallback_report(predictions, image_info)
            
        except Exception as e:
            print(f"Error generating LLM report: {e}")
            return self._generate_fallback_report(predictions, image_info)
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer specific questions about the pathology findings"""
        if not self.langchain_model and not self.groq_api_key:
            return "LLM API not configured. Please set LANGCHAIN_API_KEY or GROQ_API_KEY environment variable."
        
        try:
            prompt = f"""
            Context: {context}
            
            Question: {question}
            
            Please provide a clear, accurate answer based on the pathology findings.
            """
            
            # Try LangChain first, then fallback to Groq
            if self.langchain_model:
                response = self._call_langchain_api(prompt)
            else:
                response = self._call_groq_api(prompt)
                
            return response if response else "Unable to generate answer at this time."
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return "Error processing question. Please try again."
    
    def _call_langchain_api(self, prompt: str) -> Optional[str]:
        """Make API call using LangChain"""
        try:
            if not self.langchain_model:
                return None
                
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            with get_openai_callback() as cb:
                response = self.langchain_model.invoke(messages)
                print(f"LangChain API call - Tokens used: {cb.total_tokens}")
            
            return response.content if hasattr(response, 'content') else str(response)
                
        except Exception as e:
            print(f"LangChain API call error: {e}")
            return None
    
    def _call_groq_api(self, prompt: str) -> Optional[str]:
        """Make API call to Groq (fallback method)"""
        if not self.groq_api_key:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"Groq API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Groq API call error: {e}")
            return None
    
    def _generate_fallback_report(self, predictions: List[Dict], image_info: Dict) -> str:
        """Generate a basic report when LLM is not available"""
        cancer_patches = sum(1 for p in predictions if p.get('cancer_prob', 0) > 0.5)
        total_patches = len(predictions)
        avg_confidence = sum(p.get('cancer_prob', 0) for p in predictions) / total_patches if total_patches > 0 else 0
        cancer_rate = (cancer_patches/total_patches*100) if total_patches > 0 else 0
        
        report = f"""
        ## Histopathology Analysis Report
        
        **Analysis Summary:**
        - Total patches analyzed: {total_patches}
        - Cancerous patches detected: {cancer_patches}
        - Average confidence: {avg_confidence:.3f}
        - Cancer detection rate: {cancer_rate:.1f}%
        
        **Findings:**
        {'Cancerous tissue detected in multiple regions.' if cancer_patches > 0 else 'No cancerous tissue detected in the analyzed regions.'}
        
        **Recommendations:**
        - Review high-confidence regions for clinical correlation
        - Consider additional staining if indicated
        - Consult with pathologist for final diagnosis
        
        **Note:** This is an automated analysis. Clinical correlation is required for final diagnosis.
        """
        
        return report
    
    def get_recommendations(self, findings: str) -> str:
        """Generate clinical recommendations based on findings"""
        if not self.langchain_model and not self.groq_api_key:
            return "Review findings with a qualified pathologist for clinical recommendations."
        
        try:
            prompt = f"""
            Based on these pathology findings: {findings}
            
            Provide specific clinical recommendations for:
            1. Further diagnostic tests
            2. Treatment considerations
            3. Follow-up protocols
            4. Patient monitoring
            """
            
            # Try LangChain first, then fallback to Groq
            if self.langchain_model:
                response = self._call_langchain_api(prompt)
            else:
                response = self._call_groq_api(prompt)
                
            return response if response else "Review findings with a qualified pathologist for clinical recommendations."
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return "Review findings with a qualified pathologist for clinical recommendations."
    
    def get_status(self) -> str:
        """Get the current status of the LLM processor"""
        if self.langchain_model:
            return "✅ LangChain Connected"
        elif self.groq_api_key:
            return "✅ Groq API Available"
        else:
            return "❌ LLM Not Connected" 