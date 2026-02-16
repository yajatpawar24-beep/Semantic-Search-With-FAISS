"""
Data Processing Module for Semantic Search with FAISS

This module handles loading, cleaning, and preprocessing GitHub issues data
for semantic search applications.
"""

from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
import pandas as pd


class DataProcessor:
    """
    Handles data loading and preprocessing for GitHub issues dataset.
    
    This class provides methods to load GitHub issues, filter them,
    remove unnecessary columns, and prepare them for semantic search.
    """
    
    def __init__(self, dataset_name: str = 'lewtun/github-issues'):
        """
        Initialize the DataProcessor.
        
        Args:
            dataset_name: Name of the dataset to load from HuggingFace Hub.
                         Defaults to 'lewtun/github-issues'.
        """
        self.dataset_name = dataset_name
        self.dataset = None
        
    def load_dataset(self, split: str = 'train') -> Dataset:
        """
        Load the GitHub issues dataset.
        
        Args:
            split: Dataset split to load ('train', 'test', etc.)
        
        Returns:
            Loaded HuggingFace Dataset
            
        Example:
            >>> processor = DataProcessor()
            >>> dataset = processor.load_dataset('train')
        """
        print(f"Loading dataset: {self.dataset_name} (split: {split})")
        self.dataset = load_dataset(self.dataset_name, split=split)
        print(f"Dataset loaded: {len(self.dataset)} examples")
        return self.dataset
    
    def filter_pull_requests_and_empty_comments(self, dataset: Optional[Dataset] = None) -> Dataset:
        """
        Filter out pull requests and issues with no comments.
        
        Args:
            dataset: Dataset to filter. If None, uses self.dataset.
        
        Returns:
            Filtered dataset containing only issues with comments
            
        Example:
            >>> filtered_data = processor.filter_pull_requests_and_empty_comments()
        """
        if dataset is None:
            dataset = self.dataset
            
        print("Filtering out pull requests and empty comments...")
        filtered_dataset = dataset.filter(
            lambda x: (x['is_pull_request'] == False and len(x['comments']) > 0)
        )
        print(f"After filtering: {len(filtered_dataset)} examples")
        
        if dataset is self.dataset:
            self.dataset = filtered_dataset
            
        return filtered_dataset
    
    def remove_unnecessary_columns(
        self, 
        dataset: Optional[Dataset] = None,
        columns_to_keep: List[str] = None
    ) -> Dataset:
        """
        Remove columns not needed for semantic search.
        
        Args:
            dataset: Dataset to process. If None, uses self.dataset.
            columns_to_keep: List of column names to keep. 
                           Defaults to ['title', 'body', 'html_url', 'comments'].
        
        Returns:
            Dataset with only the specified columns
            
        Example:
            >>> cleaned_data = processor.remove_unnecessary_columns()
        """
        if dataset is None:
            dataset = self.dataset
            
        if columns_to_keep is None:
            columns_to_keep = ['title', 'body', 'html_url', 'comments']
        
        print(f"Removing unnecessary columns, keeping: {columns_to_keep}")
        columns = dataset.column_names
        columns_to_remove = set(columns).symmetric_difference(columns_to_keep)
        
        cleaned_dataset = dataset.remove_columns(columns_to_remove)
        print(f"Columns after cleaning: {cleaned_dataset.column_names}")
        
        if dataset is self.dataset:
            self.dataset = cleaned_dataset
            
        return cleaned_dataset
    
    def explode_comments(self, dataset: Optional[Dataset] = None) -> Dataset:
        """
        Explode comments so each comment gets its own row.
        
        This transforms the dataset so that each comment in an issue becomes
        a separate row, duplicating the title, body, and URL for each comment.
        
        Args:
            dataset: Dataset to process. If None, uses self.dataset.
        
        Returns:
            Dataset with exploded comments
            
        Example:
            >>> exploded_data = processor.explode_comments()
        """
        if dataset is None:
            dataset = self.dataset
            
        print("Exploding comments to individual rows...")
        dataset.set_format('pandas')
        df = dataset[:]
        
        # Explode the comments column
        comments_df = df.explode('comments', ignore_index=True)
        
        # Convert back to Dataset
        comments_dataset = Dataset.from_pandas(comments_df)
        print(f"After exploding: {len(comments_dataset)} examples")
        
        if dataset is self.dataset:
            self.dataset = comments_dataset
            
        return comments_dataset
    
    def filter_short_comments(
        self, 
        dataset: Optional[Dataset] = None,
        min_word_count: int = 15
    ) -> Dataset:
        """
        Filter out comments that are too short.
        
        Args:
            dataset: Dataset to filter. If None, uses self.dataset.
            min_word_count: Minimum number of words a comment must have.
                          Defaults to 15.
        
        Returns:
            Dataset with short comments removed
            
        Example:
            >>> filtered_data = processor.filter_short_comments(min_word_count=20)
        """
        if dataset is None:
            dataset = self.dataset
            
        print(f"Filtering comments shorter than {min_word_count} words...")
        
        # Add comment length column
        dataset = dataset.map(
            lambda x: {'comment_length': len(x['comments'].split())}
        )
        
        # Filter based on comment length
        dataset = dataset.filter(
            lambda x: x['comment_length'] > min_word_count
        )
        
        print(f"After filtering short comments: {len(dataset)} examples")
        
        if self.dataset is not None:
            self.dataset = dataset
            
        return dataset
    
    def concatenate_text_fields(self, dataset: Optional[Dataset] = None) -> Dataset:
        """
        Concatenate title, body, and comments into a single text field.
        
        This creates a new 'text' column that combines all text information
        for better semantic search results.
        
        Args:
            dataset: Dataset to process. If None, uses self.dataset.
        
        Returns:
            Dataset with added 'text' column
            
        Example:
            >>> dataset_with_text = processor.concatenate_text_fields()
        """
        if dataset is None:
            dataset = self.dataset
            
        print("Concatenating text fields...")
        
        def concatenate_text(examples):
            return {
                "text": examples["title"]
                    + " \n "
                    + examples["body"]
                    + " \n "
                    + examples["comments"]
            }
        
        dataset = dataset.map(concatenate_text)
        print("Text fields concatenated successfully")
        
        if self.dataset is not None:
            self.dataset = dataset
            
        return dataset
    
    def preprocess_full_pipeline(
        self,
        split: str = 'train',
        min_comment_length: int = 15
    ) -> Dataset:
        """
        Run the complete preprocessing pipeline.
        
        This method chains all preprocessing steps in the correct order:
        1. Load dataset
        2. Filter pull requests and empty comments
        3. Remove unnecessary columns
        4. Explode comments
        5. Filter short comments
        6. Concatenate text fields
        
        Args:
            split: Dataset split to load
            min_comment_length: Minimum word count for comments
        
        Returns:
            Fully preprocessed dataset ready for embedding generation
            
        Example:
            >>> processor = DataProcessor()
            >>> processed_data = processor.preprocess_full_pipeline()
        """
        print("=" * 60)
        print("Starting full preprocessing pipeline")
        print("=" * 60)
        
        # Step 1: Load dataset
        self.load_dataset(split=split)
        
        # Step 2: Filter pull requests and empty comments
        self.filter_pull_requests_and_empty_comments()
        
        # Step 3: Remove unnecessary columns
        self.remove_unnecessary_columns()
        
        # Step 4: Explode comments
        self.explode_comments()
        
        # Step 5: Filter short comments
        self.filter_short_comments(min_word_count=min_comment_length)
        
        # Step 6: Concatenate text fields
        self.concatenate_text_fields()
        
        print("=" * 60)
        print("Preprocessing pipeline completed!")
        print(f"Final dataset size: {len(self.dataset)} examples")
        print("=" * 60)
        
        return self.dataset


def get_dataset_sample(dataset: Dataset, n: int = 5) -> pd.DataFrame:
    """
    Get a sample of the dataset as a DataFrame for inspection.
    
    Args:
        dataset: Dataset to sample from
        n: Number of examples to return
    
    Returns:
        Pandas DataFrame with n examples
        
    Example:
        >>> sample = get_dataset_sample(dataset, n=10)
        >>> print(sample.head())
    """
    dataset.set_format('pandas')
    df = dataset[:n]
    return df
