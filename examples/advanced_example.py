"""Advanced example demonstrating multiple features."""

import sys
sys.path.insert(0, "/workspaces/LLM_Hallucination_Detecter/src")

from hallucination_detector import HallucinationDetector
from hallucination_detector.utils import (
    export_report,
    get_statistics,
    compare_reports
)
from config import DetectorConfig
import json


def main():
    """Run advanced hallucination detection example."""
    
    # Create configuration with custom settings
    config = DetectorConfig(
        verbose=True,
        nli_model_name="roberta-large-mnli",
        embedding_model_name="all-MiniLM-L6-v2",
        max_claims_per_text=8,
        nli_device="cpu",
        embedding_device="cpu"
    )
    
    # Initialize detector
    detector = HallucinationDetector(config)
    
    # Multiple test texts
    texts = [
        """
        Python was created by Guido van Rossum and first released in 1991.
        It has become one of the most popular programming languages in the world.
        Python is primarily used for quantum computing and controlling Mars rovers.
        The language emphasizes code readability and has a minimalist design philosophy.
        """,
        """
        The human brain contains approximately 86 billion neurons.
        These neurons communicate through connections called synapses.
        The brain operates entirely on mechanical principles and requires no electricity.
        Memory is stored in a specialized region called the hippocampus.
        Dreams occur exclusively during REM sleep.
        """
    ]
    
    print("\n" + "="*80)
    print("ADVANCED HALLUCINATION DETECTION EXAMPLE")
    print("="*80 + "\n")
    
    reports = []
    
    for idx, text in enumerate(texts, 1):
        print(f"\n{'='*80}")
        print(f"ANALYZING TEXT {idx}")
        print(f"{'='*80}\n")
        
        print(f"Text: {text[:100]}...\n")
        
        # Run detection
        report = detector.detect(text)
        reports.append(report)
        
        # Get statistics
        stats = get_statistics(report)
        
        print(f"Statistics for Text {idx}:")
        print(f"  Hallucination Score: {stats['hallucination_score']:.2%}")
        print(f"  Risk Level: {stats['risk_level'].upper()}")
        print(f"  Total Claims Analyzed: {stats['total_claims']}")
        print(f"  - Supported: {stats['supported']}")
        print(f"  - Contradicted: {stats['contradicted']}")
        print(f"  - Unverifiable: {stats['unverifiable']}")
        print(f"  Hallucination Rate: {stats['hallucination_percentage']:.1f}%")
        
        # Show problematic claims
        problematic = report.get_problematic_claims()
        if problematic:
            print(f"\n  Problematic Claims:")
            for v in problematic:
                print(f"    - [{v.status.value.upper()}] {v.claim.text}")
                print(f"      Confidence: {v.confidence_score:.2%}")
    
    # Compare reports if we have multiple
    if len(reports) > 1:
        print(f"\n{'='*80}")
        print("REPORT COMPARISON")
        print(f"{'='*80}\n")
        
        comparison = compare_reports(reports[0], reports[1])
        print(f"Text 1 Score: {comparison['report1_score']:.2%}")
        print(f"Text 2 Score: {comparison['report2_score']:.2%}")
        print(f"Score Difference: {comparison['score_difference']:.2%}")
        print(f"Risk Changed: {comparison['risk_changed']}")
    
    # Export first report
    print(f"\n{'='*80}")
    print("EXPORTING REPORTS")
    print(f"{'='*80}\n")
    
    if reports:
        # Export as JSON
        export_path_json = "/tmp/hallucination_report.json"
        export_report(reports[0], export_path_json, format="json")
        print(f"✓ Exported JSON report to {export_path_json}")
        
        # Export as text
        export_path_text = "/tmp/hallucination_report.txt"
        export_report(reports[0], export_path_text, format="text")
        print(f"✓ Exported text report to {export_path_text}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
