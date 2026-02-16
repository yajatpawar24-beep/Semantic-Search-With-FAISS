#!/usr/bin/env python3
"""
Inference Utilities for Semantic Search

This module provides easy-to-use inference utilities for semantic search,
including model loading, query processing, and result formatting.
"""

import os
import pickle
from typing import List, Dict, Optional, Union
import pandas as pd
from pathlib import Path

from embeddings import EmbeddingGenerator
from semantic_search import SemanticSearchEngine
from datasets import Dataset


class SearchInference:
    """
    High-level inference class for semantic search.
    
    This class provides a simple API for loading saved search engines
    and performing queries.
    """
    
    def __init__(
        self,
        model_checkpoint: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_checkpoint: Model checkpoint to use for embeddings
            
        Example:
            >>> inference = SearchInference()
        """
        self.model_checkpoint = model_checkpoint
        self.search_engine = None
        self.embedding_generator = None
        
    def load_or_build_search_engine(
        self,
        dataset_path: Optional[str] = None,
        dataset_name: str = 'lewtun/github-issues',
        force_rebuild: bool = False
    ) -> SemanticSearchEngine:
        """
        Load a saved search engine or build a new one.
        
        Args:
            dataset_path: Path to saved dataset (if available)
            dataset_name: HuggingFace dataset name (if building from scratch)
            force_rebuild: Force rebuild even if saved dataset exists
        
        Returns:
            Ready-to-use SemanticSearchEngine
            
        Example:
            >>> inference = SearchInference()
            >>> search_engine = inference.load_or_build_search_engine()
        """
        from semantic_search import create_search_engine_from_scratch
        
        if dataset_path and os.path.exists(dataset_path) and not force_rebuild:
            print(f"Loading dataset from {dataset_path}...")
            dataset = Dataset.load_from_disk(dataset_path)
            
            self.embedding_generator = EmbeddingGenerator(self.model_checkpoint)
            self.search_engine = SemanticSearchEngine(self.embedding_generator)
            self.search_engine.build_index(dataset)
            
        else:
            print("Building search engine from scratch...")
            self.search_engine = create_search_engine_from_scratch(
                dataset_name=dataset_name,
                model_checkpoint=self.model_checkpoint
            )
        
        return self.search_engine
    
    def search(
        self,
        query: str,
        k: int = 5,
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, Dict]:
        """
        Perform a semantic search query.
        
        Args:
            query: Search query string
            k: Number of results to return
            return_dataframe: Whether to return results as DataFrame
        
        Returns:
            Search results as DataFrame or dictionary
            
        Example:
            >>> results = inference.search("How to load datasets?", k=5)
        """
        if self.search_engine is None:
            raise RuntimeError("Search engine not loaded. Call load_or_build_search_engine() first.")
        
        if return_dataframe:
            return self.search_engine.search_and_format(query, k=k)
        else:
            scores, samples = self.search_engine.search(query, k=k)
            return {'scores': scores, 'samples': samples}
    
    def search_multiple(
        self,
        queries: List[str],
        k: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Search for multiple queries at once.
        
        Args:
            queries: List of query strings
            k: Number of results per query
        
        Returns:
            Dictionary mapping queries to result DataFrames
            
        Example:
            >>> queries = ["How to train?", "How to save model?"]
            >>> results = inference.search_multiple(queries, k=3)
        """
        if self.search_engine is None:
            raise RuntimeError("Search engine not loaded. Call load_or_build_search_engine() first.")
        
        results = {}
        for query in queries:
            results[query] = self.search_engine.search_and_format(query, k=k)
        
        return results
    
    def get_top_result(self, query: str) -> Dict:
        """
        Get only the top result for a query.
        
        Args:
            query: Search query string
        
        Returns:
            Dictionary containing the top result
            
        Example:
            >>> top_result = inference.get_top_result("How to use datasets?")
            >>> print(top_result['title'])
        """
        results_df = self.search(query, k=1, return_dataframe=True)
        return results_df.iloc[0].to_dict()
    
    def save_dataset(self, save_path: str) -> None:
        """
        Save the indexed dataset to disk for faster loading later.
        
        Args:
            save_path: Directory path to save the dataset
            
        Example:
            >>> inference.save_dataset("./saved_index")
        """
        if self.search_engine is None or self.search_engine.dataset is None:
            raise RuntimeError("No dataset to save. Build search engine first.")
        
        print(f"Saving dataset to {save_path}...")
        self.search_engine.dataset.save_to_disk(save_path)
        print("Dataset saved successfully!")


class QuickSearch:
    """
    Quick one-liner semantic search utility.
    
    This class provides the simplest possible interface for semantic search.
    """
    
    @staticmethod
    def search(
        query: str,
        k: int = 5,
        model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        dataset: str = 'lewtun/github-issues'
    ) -> pd.DataFrame:
        """
        Perform a quick semantic search with one function call.
        
        This creates a search engine from scratch and performs the search.
        Use this for quick experiments or one-off searches.
        
        Args:
            query: Search query string
            k: Number of results to return
            model: Model checkpoint to use
            dataset: Dataset name to search
        
        Returns:
            DataFrame with search results
            
        Example:
            >>> results = QuickSearch.search(
            ...     "How to load a dataset offline?",
            ...     k=5
            ... )
        """
        from semantic_search import create_search_engine_from_scratch
        
        print("Creating search engine (this may take a few minutes)...")
        search_engine = create_search_engine_from_scratch(
            dataset_name=dataset,
            model_checkpoint=model
        )
        
        return search_engine.search_and_format(query, k=k)


def simple_search_example():
    """
    Example of the simplest way to perform semantic search.
    
    This function demonstrates how to use the QuickSearch utility
    for one-off searches.
    """
    print("\n" + "="*80)
    print("SIMPLE SEARCH EXAMPLE")
    print("="*80 + "\n")
    
    query = "How can I load a dataset offline?"
    print(f"Query: {query}\n")
    
    results = QuickSearch.search(query, k=3)
    
    print("\nTop 3 Results:")
    print("-" * 80)
    for idx, row in results.iterrows():
        print(f"\nResult #{idx + 1}")
        print(f"Score: {row['scores']:.4f}")
        print(f"Title: {row['title']}")
        print(f"URL: {row['html_url']}")
        print(f"Comment Preview: {row['comments'][:150]}...")


def advanced_search_example():
    """
    Example of advanced search with custom configuration.
    
    This function demonstrates how to use the SearchInference class
    for more control over the search process.
    """
    print("\n" + "="*80)
    print("ADVANCED SEARCH EXAMPLE")
    print("="*80 + "\n")
    
    # Initialize inference engine
    inference = SearchInference(
        model_checkpoint="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    
    # Build or load search engine
    search_engine = inference.load_or_build_search_engine()
    
    # Multiple queries
    queries = [
        "How to fine-tune BERT?",
        "What are the best practices for tokenization?",
        "How to handle out of memory errors?"
    ]
    
    print("Running multiple queries...\n")
    results = inference.search_multiple(queries, k=2)
    
    for query, result_df in results.items():
        print(f"\nQuery: {query}")
        print("-" * 80)
        print(result_df[['title', 'scores']].to_string(index=False))
        print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Search Inference Examples")
    parser.add_argument(
        '--example',
        type=str,
        choices=['simple', 'advanced'],
        default='simple',
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    if args.example == 'simple':
        simple_search_example()
    elif args.example == 'advanced':
        advanced_search_example()
