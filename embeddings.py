"""
Embeddings Generation Module for Semantic Search

This module handles generating embeddings for text using sentence transformers
and provides utilities for efficient batch processing.
"""

from typing import List, Union, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from tqdm import tqdm


class EmbeddingGenerator:
    """
    Generates text embeddings using sentence transformer models.
    
    This class handles model loading, embedding generation, and batch processing
    for creating semantic embeddings from text data.
    """
    
    def __init__(
        self,
        model_checkpoint: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        device: Optional[str] = None
    ):
        """
        Initialize the embedding generator with a pretrained model.
        
        Args:
            model_checkpoint: HuggingFace model checkpoint to use.
                            Defaults to multi-qa-mpnet-base-dot-v1.
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect).
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> # Or with custom model:
            >>> generator = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")
        """
        self.model_checkpoint = model_checkpoint
        
        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initializing embedding generator...")
        print(f"Model: {model_checkpoint}")
        print(f"Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print("Model loaded successfully!")
    
    @staticmethod
    def cls_pooling(model_output):
        """
        Perform CLS token pooling on model output.
        
        This extracts the embedding from the [CLS] token (first token)
        of the model output, which represents the entire sequence.
        
        Args:
            model_output: Output from the transformer model
        
        Returns:
            Tensor containing the CLS token embeddings
        """
        return model_output.last_hidden_state[:, 0]
    
    def get_embeddings(
        self,
        text_list: Union[str, List[str]],
        normalize: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            text_list: Single text string or list of text strings
            normalize: Whether to L2-normalize the embeddings
        
        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim)
            
        Example:
            >>> embeddings = generator.get_embeddings(["Hello world", "Test text"])
            >>> print(embeddings.shape)  # (2, 768)
        """
        # Handle single string input
        if isinstance(text_list, str):
            text_list = [text_list]
        
        # Tokenize the input
        encoded_input = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self.cls_pooling(model_output)
        
        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def generate_embeddings_for_dataset(
        self,
        dataset: Dataset,
        text_column: str = 'text',
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dataset:
        """
        Generate embeddings for an entire dataset with batch processing.
        
        Args:
            dataset: HuggingFace Dataset to generate embeddings for
            text_column: Name of the column containing text
            batch_size: Batch size for processing (higher = faster but more memory)
            show_progress: Whether to show a progress bar
        
        Returns:
            Dataset with added 'embeddings' column
            
        Example:
            >>> dataset_with_embeddings = generator.generate_embeddings_for_dataset(
            ...     dataset, batch_size=64
            ... )
        """
        print(f"Generating embeddings for {len(dataset)} examples...")
        print(f"Batch size: {batch_size}")
        
        def embed_batch(examples):
            """Helper function to embed a batch of examples."""
            embeddings = self.get_embeddings(examples[text_column])
            return {'embeddings': embeddings.tolist()}
        
        # Generate embeddings in batches
        dataset_with_embeddings = dataset.map(
            embed_batch,
            batched=True,
            batch_size=batch_size
        )
        
        print("Embeddings generated successfully!")
        return dataset_with_embeddings
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text query.
        
        Convenience method for embedding single queries for search.
        
        Args:
            text: Text string to embed
        
        Returns:
            NumPy array of shape (1, embedding_dim)
            
        Example:
            >>> query_embedding = generator.embed_single_text("How to load a dataset?")
        """
        return self.get_embeddings([text])
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Integer representing the embedding dimension
            
        Example:
            >>> dim = generator.get_embedding_dimension()
            >>> print(f"Embedding dimension: {dim}")  # 768
        """
        # Generate a dummy embedding to get dimension
        dummy_embedding = self.get_embeddings(["test"])
        return dummy_embedding.shape[1]


def compare_embeddings_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    metric: str = 'cosine'
) -> float:
    """
    Compute similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metric: Similarity metric to use ('cosine' or 'euclidean')
    
    Returns:
        Similarity score (higher is more similar for cosine)
        
    Example:
        >>> emb1 = generator.get_embeddings(["cat"])
        >>> emb2 = generator.get_embeddings(["dog"])
        >>> similarity = compare_embeddings_similarity(emb1[0], emb2[0])
    """
    if metric == 'cosine':
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)
    elif metric == 'euclidean':
        # Euclidean distance (lower is more similar)
        distance = np.linalg.norm(embedding1 - embedding2)
        return float(distance)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'.")
