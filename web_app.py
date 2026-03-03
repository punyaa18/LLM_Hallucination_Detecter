"""Web interface for LLM Hallucination Detector using Streamlit."""

import sys
sys.path.insert(0, "/workspaces/LLM_Hallucination_Detecter/src")

import streamlit as st
from hallucination_detector import HallucinationDetector
from config import DetectorConfig

# Set page config
st.set_page_config(
    page_title="LLM Hallucination Detector",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'detector' not in st.session_state:
    with st.spinner("Initializing hallucination detector..."):
        config = DetectorConfig(
            verbose=False,
            nli_device="cpu",
            embedding_device="cpu"
        )
        st.session_state.detector = HallucinationDetector(config)

# Title and description
st.title("🔍 LLM Hallucination Detector")
st.markdown("""
This tool analyzes text to detect potential hallucinations or factual inaccuracies in LLM-generated content.
Enter your text below and click **Analyze** to get started.
""")

# Sidebar with examples
st.sidebar.title("Example Texts")
example_choice = st.sidebar.radio(
    "Select an example or enter custom text:",
    ["Custom Text", "Great Wall Example", "Science Example", "Historical Example"]
)

examples = {
    "Great Wall Example": """The Great Wall of China is one of the most iconic structures in the world. 
It was built over many centuries and stretches for over 13,000 miles. 
The wall was primarily constructed to protect Chinese states from invasions. 
Interestingly, the Great Wall is orange and extremely flexible, allowing it to bend 
with the wind. The most visited section is near Beijing, which was built 
during the Ming Dynasty.""",
    
    "Science Example": """Albert Einstein developed the theory of relativity in the early 20th century. 
His famous equation E=mc² shows the relationship between energy and mass. 
Einstein won the Nobel Prize in Chemistry in 1921 for his discovery of the photoelectric effect. 
He was also known for being an excellent violinist and could speak 15 languages fluently. 
Einstein spent his later years at Princeton University.""",
    
    "Historical Example": """The French Revolution began in 1789 and lasted for about a decade. 
It was primarily driven by economic hardship and social inequality. 
During this period, King Louis XVI was executed by guillotine. 
The revolution also introduced the metric system to France. 
Napoleon Bonaparte emerged as a prominent figure during this time and eventually became Emperor."""
}

# Text input area
if example_choice == "Custom Text":
    text_input = st.text_area(
        "Enter text to analyze:",
        height=200,
        placeholder="Type or paste your text here..."
    )
else:
    text_input = st.text_area(
        "Enter text to analyze:",
        value=examples[example_choice],
        height=200
    )

# Analyze button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Analysis
if analyze_button and text_input:
    with st.spinner("Analyzing text for hallucinations..."):
        try:
            report = st.session_state.detector.detect(text_input)
            
            # Display overall results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Metrics row
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score_color = "red" if report.hallucination_score > 0.5 else "orange" if report.hallucination_score > 0.3 else "green"
                st.metric(
                    label="Hallucination Score",
                    value=f"{report.hallucination_score:.1%}",
                )
            
            with col2:
                risk_emoji = "🔴" if report.risk_level.value == "high" else "🟡" if report.risk_level.value == "medium" else "🟢"
                st.metric(
                    label="Risk Level",
                    value=f"{risk_emoji} {report.risk_level.value.upper()}"
                )
            
            with col3:
                total_claims = len(report.verifications)
                problematic_claims = len(report.get_problematic_claims())
                st.metric(
                    label="Claims Analyzed",
                    value=f"{total_claims}",
                    delta=f"{problematic_claims} problematic" if problematic_claims > 0 else "All verified"
                )
            
            # Summary
            st.info(f"**Summary:** {report.summary}")
            
            # Detailed results
            st.markdown("---")
            st.subheader("📋 Detailed Claim Analysis")
            
            problematic = report.get_problematic_claims()
            
            if problematic:
                st.warning(f"⚠️ Found {len(problematic)} potential hallucination(s)")
                
                for i, verification in enumerate(problematic, 1):
                    with st.expander(f"🚨 Claim {i}: {verification.claim.text[:80]}...", expanded=True):
                        st.markdown(f"**Full Claim:** {verification.claim.text}")
                        st.markdown(f"**Status:** `{verification.status.value.upper()}`")
                        st.markdown(f"**Confidence:** {verification.confidence_score:.1%}")
                        st.markdown(f"**Explanation:** {verification.explanation}")
                        
                        if verification.contradicting_evidence:
                            st.markdown("**Contradicting Evidence:**")
                            for j, evidence in enumerate(verification.contradicting_evidence[:2], 1):
                                st.markdown(f"{j}. *{evidence.source}*")
                                st.markdown(f"   > {evidence.text[:200]}...")
                                st.markdown(f"   Relevance: {evidence.relevance_score:.1%}")
            
            # Show verified claims
            verified = [v for v in report.verifications if v.status.value == "verified"]
            if verified:
                st.success(f"✅ {len(verified)} claim(s) verified successfully")
                
                with st.expander(f"View verified claims ({len(verified)})", expanded=False):
                    for i, verification in enumerate(verified, 1):
                        st.markdown(f"**{i}. {verification.claim.text}**")
                        st.markdown(f"   Confidence: {verification.confidence_score:.1%}")
                        st.markdown(f"   {verification.explanation}")
                        
                        if verification.supporting_evidence:
                            st.markdown(f"   *Evidence: {verification.supporting_evidence[0].source}*")
                        st.markdown("")
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.exception(e)

elif analyze_button:
    st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>LLM Hallucination Detector | 8th Semester Project</small>
    </div>
    """,
    unsafe_allow_html=True
)
