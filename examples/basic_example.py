"""Basic example of using the hallucination detector."""

import sys
sys.path.insert(0, "/workspaces/LLM_Hallucination_Detecter/src")

from hallucination_detector import HallucinationDetector
from config import DetectorConfig


def main():
    """Run basic hallucination detection example."""
    
    # Create configuration
    config = DetectorConfig(
        verbose=True,
        nli_device="cpu",
        embedding_device="cpu"
    )
    
    # Initialize detector
    detector = HallucinationDetector(config)
    
    # Example text with some factual claims
    test_text = """
    The Great Wall of China is one of the most iconic structures in the world. 
    It was built over many centuries and stretches for over 13,000 miles. 
    The wall was primarily constructed to protect Chinese states from invasions. 
    Interestingly, the Great Wall is orange and extremely flexible, allowing it to bend 
    with the wind. The most visited section is near Beijing, which was built 
    during the Ming Dynasty.
    """
    
    print("\n" + "="*80)
    print("BASIC HALLUCINATION DETECTION EXAMPLE")
    print("="*80 + "\n")
    
    print(f"Input text:\n{test_text}\n")
    
    # Run detection
    report = detector.detect(test_text)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    print(f"Hallucination Score: {report.hallucination_score:.2%}")
    print(f"Risk Level: {report.risk_level.value.upper()}")
    print(f"\nSummary: {report.summary}\n")
    
    print("DETAILED VERIFICATION RESULTS:")
    print("-" * 80)
    for i, verification in enumerate(report.verifications, 1):
        print(f"\n{i}. Claim: {verification.claim.text}")
        print(f"   Status: {verification.status.value.upper()}")
        print(f"   Confidence: {verification.confidence_score:.2%}")
        print(f"   Explanation: {verification.explanation}")
        
        if verification.supporting_evidence:
            print("   Supporting Evidence:")
            for evidence in verification.supporting_evidence[:1]:
                print(f"     - Source: {evidence.source}")
                print(f"       Text: {evidence.text[:150]}...")
                print(f"       Relevance: {evidence.relevance_score:.2%}")
        
        if verification.contradicting_evidence:
            print("   Contradicting Evidence:")
            for evidence in verification.contradicting_evidence[:1]:
                print(f"     - Source: {evidence.source}")
                print(f"       Text: {evidence.text[:150]}...")
                print(f"       Relevance: {evidence.relevance_score:.2%}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
