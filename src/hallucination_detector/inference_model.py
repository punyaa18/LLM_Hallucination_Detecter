"""Module for NLI and semantic similarity inference."""

import torch
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

from config.config import InferenceConfig


class InferenceModel:
    """Performs Natural Language Inference between claims and evidence."""
    
    def __init__(
        self,
        nli_model_name: str = "roberta-large-mnli",
        device: str = "cpu",
        config: InferenceConfig = None
    ):
        """Initialize inference model.
        
        Args:
            nli_model_name: Name of the NLI model.
            device: Device to run model on (cpu or cuda).
            config: Configuration for inference.
        """
        self.device = device
        self.config = config or InferenceConfig()
        
        # Load NLI model
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli_model.to(device)
        self.nli_model.eval()
        
        # Load semantic similarity model
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    def check_nli(
        self,
        premise: str,
        hypothesis: str
    ) -> Tuple[str, float]:
        """Check Natural Language Inference between premise and hypothesis.
        
        Args:
            premise: The premise (evidence).
            hypothesis: The hypothesis (claim).
            
        Returns:
            Tuple of (inference_label, confidence_score).
        """
        # Prepare input
        input_ids = self.nli_tokenizer.encode(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get logits
        with torch.no_grad():
            logits = self.nli_model(input_ids)[0]
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        scores = probs[0].cpu().numpy()
        
        # Get prediction
        pred_idx = scores.argmax()
        confidence = float(scores[pred_idx])
        label = self.config.label_mapping.get(pred_idx, "unknown")
        
        return label, confidence
    
    def check_semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Check semantic similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        embeddings = self.sim_model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return float(similarity[0][0])
    
    def batch_nli(
        self,
        premises: List[str],
        hypotheses: List[str]
    ) -> List[Tuple[str, float]]:
        """Perform batch NLI inference.
        
        Args:
            premises: List of premises.
            hypotheses: List of hypotheses.
            
        Returns:
            List of (label, confidence) tuples.
        """
        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses must have same length")
        
        results = []
        for premise, hypothesis in zip(premises, hypotheses):
            result = self.check_nli(premise, hypothesis)
            results.append(result)
        
        return results
    
    def batch_semantic_similarity(
        self,
        texts1: List[str],
        texts2: List[str]
    ) -> List[float]:
        """Perform batch semantic similarity checks.
        
        Args:
            texts1: First list of texts.
            texts2: Second list of texts.
            
        Returns:
            List of similarity scores.
        """
        if len(texts1) != len(texts2):
            raise ValueError("Text lists must have same length")
        
        embeddings1 = self.sim_model.encode(texts1, convert_to_tensor=True)
        embeddings2 = self.sim_model.encode(texts2, convert_to_tensor=True)
        
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            sim = util.pytorch_cos_sim(emb1, emb2)
            similarities.append(float(sim[0][0]))
        
        return similarities
