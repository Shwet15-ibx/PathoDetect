"""
Report Generator Component for PathoDetect+
Generates LLM-assisted pathology reports
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List

class ReportGenerator:
    def __init__(self, config):
        self.config = config

    def generate_report(self, predictions: List[Dict], image, llm_processor) -> str:
        """Generate a comprehensive pathology report"""
        try:
            # Prepare image info
            image_info = {
                'filename': getattr(image, 'name', 'Unknown'),
                'size': getattr(image, 'size', 'Unknown'),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Generate report using LLM
            report = llm_processor.generate_report(predictions, image_info)
            return report
            
        except Exception as e:
            st.error(f"Error generating report: {e}")
            return self._generate_basic_report(predictions)
    
    def _generate_basic_report(self, predictions: List[Dict]) -> str:
        """Generate a basic report when LLM is not available"""
        cancer_patches = sum(1 for p in predictions if p.get('cancer_prob', 0) > 0.5)
        total_patches = len(predictions)
        avg_confidence = sum(p.get('cancer_prob', 0) for p in predictions) / total_patches if total_patches > 0 else 0
        
        report = f"""
        ## Basic Pathology Report
        
        **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        **Summary:**
        - Total patches analyzed: {total_patches}
        - Cancerous patches detected: {cancer_patches}
        - Average confidence: {avg_confidence:.3f}
        
        **Findings:**
        {'Cancerous tissue detected in multiple regions.' if cancer_patches > 0 else 'No cancerous tissue detected in the analyzed regions.'}
        
        **Note:** This is a basic automated analysis. For detailed LLM-assisted reporting, please configure the API keys.
        """
        
        return report
    
    def display_interactive_qa(self, predictions: List[Dict], llm_processor) -> None:
        """Display interactive Q&A section"""
        st.subheader("🔍 Interactive Q&A")
        st.write("Ask questions about the pathology findings:")
        
        # Prepare context
        cancer_patches = sum(1 for p in predictions if p.get('cancer_prob', 0) > 0.5)
        total_patches = len(predictions)
        avg_confidence = sum(p.get('cancer_prob', 0) for p in predictions) / total_patches if total_patches > 0 else 0
        cancer_rate = (cancer_patches/total_patches*100) if total_patches > 0 else 0
        
        context = f"""
        Analysis Results:
        - Total patches: {total_patches}
        - Cancer patches: {cancer_patches}
        - Average confidence: {avg_confidence:.3f}
        - Cancer rate: {cancer_rate:.1f}%
        """
        
        # Question input
        question = st.text_input("Enter your question:", 
                                placeholder="e.g., What type of cancer is most likely?")
        
        if st.button("Ask Question") and question:
            with st.spinner("Getting answer..."):
                answer = llm_processor.answer_question(question, context)
                
                # Display Q&A
                st.markdown("---")
                st.markdown(f"**Q:** {question}")
                st.markdown(f"**A:** {answer}")
        
        # Pre-defined questions
        st.subheader("💡 Suggested Questions")
        suggested_questions = [
            "What are the key findings in this analysis?",
            "What type of breast cancer is most likely present?",
            "What are the clinical implications of these results?",
            "What additional tests should be considered?",
            "What is the confidence level of this analysis?"
        ]
        
        for i, q in enumerate(suggested_questions):
            if st.button(f"Q{i+1}: {q}", key=f"suggested_q_{i}"):
                with st.spinner("Getting answer..."):
                    answer = llm_processor.answer_question(q, context)
                    st.markdown(f"**A:** {answer}") 