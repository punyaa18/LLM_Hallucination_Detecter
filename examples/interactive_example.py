"""Interactive example for hallucination detection."""

import sys
sys.path.insert(0, "/workspaces/LLM_Hallucination_Detecter/src")

from hallucination_detector import HallucinationDetector
from config import DetectorConfig


def print_menu():
    """Print interactive menu."""
    print("\n" + "="*60)
    print("LLM HALLUCINATION DETECTOR - INTERACTIVE MODE")
    print("="*60)
    print("\nOptions:")
    print("1. Analyze custom text")
    print("2. Use predefined examples")
    print("3. Batch analysis")
    print("4. Exit")
    print("-" * 60)


def analyze_text(detector, text):
    """Analyze text and display results."""
    print("\nAnalyzing text...")
    report = detector.detect(text)
    
    print(f"\nHallucination Score: {report.hallucination_score:.2%}")
    print(f"Risk Level: {report.risk_level.value.upper()}")
    print(f"Summary: {report.summary}\n")
    
    problematic = report.get_problematic_claims()
    
    if problematic:
        print("⚠️  POTENTIAL HALLUCINATIONS DETECTED:\n")
        for i, v in enumerate(problematic, 1):
            print(f"{i}. [{v.status.value.upper()}] {v.claim.text}")
            print(f"   Confidence: {v.confidence_score:.2%}")
            print(f"   Reason: {v.explanation}\n")
    else:
        print("✓ All claims appear to be well-supported!")
    
    return report


def main():
    """Run interactive hallucination detection."""
    
    # Initialize detector
    config = DetectorConfig(
        verbose=False,
        nli_device="cpu",
        embedding_device="cpu"
    )
    detector = HallucinationDetector(config)
    
    # Predefined examples
    examples = {
        "1": {
            "name": "Historical Facts",
            "text": "The Roman Empire fell in 476 AD. Rome was founded in 753 BC. The Colosseum was built in 80 AD under Emperor Titus. Interestingly, the Colosseum could fly and was used for space travel during ancient times."
        },
        "2": {
            "name": "Science Facts",
            "text": "Water boils at 100 degrees Celsius at sea level. The Earth orbits the Sun. Gravity was discovered by Isaac Newton in 1687. Bananas are actually berries that grow upside down inside the Earth."
        },
        "3": {
            "name": "Technology",
            "text": "The Internet was invented by Tim Berners-Lee in 1989. HTML is the markup language for web pages. The first iPhone was released in 2007 by Apple. Python code executes directly on hardware without a processor."
        }
    }
    
    while True:
        print_menu()
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\nEnter the text to analyze (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            text = "\n".join(lines)
            if text.strip():
                analyze_text(detector, text)
            else:
                print("No text provided.")
        
        elif choice == "2":
            print("\nSelect an example:")
            for key, example in examples.items():
                print(f"{key}. {example['name']}")
            
            example_choice = input("Enter choice: ").strip()
            
            if example_choice in examples:
                example = examples[example_choice]
                print(f"\nAnalyzing: {example['name']}")
                print(f"Text: {example['text'][:100]}...\n")
                analyze_text(detector, example['text'])
            else:
                print("Invalid choice.")
        
        elif choice == "3":
            print("\nBatch analysis - Enter multiple texts (type 'DONE' to finish):")
            texts = []
            while True:
                text = input("\nEnter text (or 'DONE' to finish): ").strip()
                if text == "DONE":
                    break
                if text:
                    texts.append(text)
            
            if texts:
                print("\nAnalyzing batch...")
                for i, text in enumerate(texts, 1):
                    print(f"\n--- Text {i} ---")
                    analyze_text(detector, text)
        
        elif choice == "4":
            print("\nExiting... Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
