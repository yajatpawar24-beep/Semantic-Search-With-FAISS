"""
Semantic Search Module using FAISS

This module provides semantic search capabilities using FAISS indexing
for efficient similarity search over large document collections.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from datasets import Dataset
from embeddings import EmbeddingGenerator


class SemanticSearchEngine:
    """
    Semantic search engine using FAISS for efficient similarity search.
    
    This class handles FAISS index creation, query processing, and result retrieval
    for semantic search over document collections.
    """
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        index_column: str = 'embeddings'
    ):
        """
        Initialize the semantic search engine.
        
        Args:
            embedding_generator: EmbeddingGenerator instance for query encoding.
                               If None, creates a new instance.
            index_column: Name of the column containing embeddings in the dataset.
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> search_engine = SemanticSearchEngine(generator)
        """
        self.index_column = index_column
        self.dataset = None
        self.index_built = False
        
        # Create or use provided embedding generator
        if embedding_generator is None:
            print("No embedding generator provided, creating default...")
            self.embedding_generator = EmbeddingGenerator()
        else:
            self.embedding_generator = embedding_generator
    
    def build_index(self, dataset: Dataset) -> None:
        """
        Build FAISS index for the dataset.
        
        This creates an efficient similarity search index using FAISS.
        The dataset must already have embeddings in the specified column.
        
        Args:
            dataset: Dataset with embeddings column
        
        Raises:
            ValueError: If the dataset doesn't have the embeddings column
            
        Example:
            >>> search_engine.build_index(dataset_with_embeddings)
        """
        if self.index_column not in dataset.column_names:
            raise ValueError(
                f"Dataset must have '{self.index_column}' column. "
                f"Available columns: {dataset.column_names}"
            )
        
        print(f"Building FAISS index on column '{self.index_column}'...")
        print(f"Dataset size: {len(dataset)} documents")
        
        # Add FAISS index to the dataset
        dataset.add_faiss_index(column=self.index_column)
        
        self.dataset = dataset
        self.index_built = True
        
        print("FAISS index built successfully!")
    
    def search(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Search for the most relevant documents for a query.
        
        Args:
            query: Text query to search for
            k: Number of results to return
            return_scores: Whether to return similarity scores
        
        Returns:
            Tuple of (scores, samples) where:
                - scores: Array of similarity scores for each result
                - samples: Dictionary containing the retrieved documents
        
        Raises:
            RuntimeError: If index hasn't been built yet
            
        Example:
            >>> scores, results = search_engine.search(
            ...     "How can I load a dataset offline?",
            ...     k=5
            ... )
        """
        if not self.index_built:
            raise RuntimeError(
                "Index has not been built yet. Call build_index() first."
            )
        
        print(f"Searching for: '{query}'")
        print(f"Retrieving top {k} results...")
        
        # Generate embedding for the query
        query_embedding = self.embedding_generator.embed_single_text(query)
        
        # Search using FAISS
        scores, samples = self.dataset.get_nearest_examples(
            self.index_column,
            query_embedding,
            k=k
        )
        
        return scores, samples
    
    def search_and_format(
        self,
        query: str,
        k: int = 5,
        sort_by_score: bool = True
    ) -> pd.DataFrame:
        """
        Search and return results as a formatted DataFrame.
        
        Args:
            query: Text query to search for
            k: Number of results to return
            sort_by_score: Whether to sort results by score (descending)
        
        Returns:
            Pandas DataFrame with search results and scores
            
        Example:
            >>> results_df = search_engine.search_and_format(
            ...     "How to use transformers?",
            ...     k=10
            ... )
            >>> print(results_df[['title', 'scores']].head())
        """
        scores, samples = self.search(query, k=k)
        
        # Convert to DataFrame
        results_df = pd.DataFrame.from_dict(samples)
        results_df['scores'] = scores
        
        # Sort by scores if requested
        if sort_by_score:
            results_df.sort_values('scores', ascending=False, inplace=True)
        
        return results_df
    
    def display_results(
        self,
        query: str,
        k: int = 5,
        show_full_text: bool = False
    ) -> None:
        """
        Search and display results in a readable format.
        
        Args:
            query: Text query to search for
            k: Number of results to return
            show_full_text: Whether to show the full concatenated text
        
        Example:
            >>> search_engine.display_results(
            ...     "How can I load a dataset offline?",
            ...     k=3
            ... )
        """
        results_df = self.search_and_format(query, k=k)
        
        print("\n" + "=" * 80)
        print(f"SEARCH RESULTS FOR: '{query}'")
        print("=" * 80 + "\n")
        
        for idx, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"RESULT #{idx}")
            print(f"Score: {row.scores:.4f}")
            print(f"Title: {row.title}")
            
            # Show comment (truncate if too long)
            comment = row.comments
            if len(comment) > 300 and not show_full_text:
                comment = comment[:300] + "..."
            print(f"Comment: {comment}")
            
            print(f"URL: {row.html_url}")
            print("-" * 80 + "\n")
    
    def batch_search(
        self,
        queries: List[str],
        k: int = 5
    ) -> List[pd.DataFrame]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of query strings
            k: Number of results to return for each query
        
        Returns:
            List of DataFrames, one for each query
            
        Example:
            >>> queries = [
            ...     "How to load a dataset?",
            ...     "How to train a model?",
            ...     "How to save model weights?"
            ... ]
            >>> results = search_engine.batch_search(queries, k=3)
        """
        print(f"Performing batch search for {len(queries)} queries...")
        
        results = []
        for query in queries:
            result_df = self.search_and_format(query, k=k)
            results.append(result_df)
        
        print("Batch search completed!")
        return results
    
    def get_similar_documents(
        self,
        document_index: int,
        k: int = 5
    ) -> pd.DataFrame:
        """
        Find documents similar to a specific document in the dataset.
        
        Args:
            document_index: Index of the document to find similar documents for
            k: Number of similar documents to return (excluding the document itself)
        
        Returns:
            DataFrame with similar documents
            
        Example:
            >>> similar_docs = search_engine.get_similar_documents(
            ...     document_index=42,
            ...     k=5
            ... )
        """
        if not self.index_built:
            raise RuntimeError(
                "Index has not been built yet. Call build_index() first."
            )
        
        # Get the embedding of the document
        self.dataset.set_format('numpy')
        doc_embedding = self.dataset[document_index][self.index_column]
        doc_embedding = doc_embedding.reshape(1, -1)
        
        # Search for similar documents (k+1 because the document itself will be included)
        scores, samples = self.dataset.get_nearest_examples(
            self.index_column,
            doc_embedding,
            k=k+1
        )
        
        # Convert to DataFrame and remove the query document itself
        results_df = pd.DataFrame.from_dict(samples)
        results_df['scores'] = scores
        results_df = results_df.iloc[1:]  # Skip first result (the document itself)
        
        return results_df


def create_search_engine_from_scratch(
    dataset_name: str = 'lewtun/github-issues',
    model_checkpoint: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    min_comment_length: int = 15,
    batch_size: int = 32
) -> SemanticSearchEngine:
    """
    Create a semantic search engine from scratch with full pipeline.
    
    This function handles the complete workflow:
    1. Load and preprocess data
    2. Generate embeddings
    3. Build FAISS index
    
    Args:
        dataset_name: Name of the dataset to load
        model_checkpoint: Model to use for embeddings
        min_comment_length: Minimum word count for comments
        batch_size: Batch size for embedding generation
    
    Returns:
        Ready-to-use SemanticSearchEngine instance
        
    Example:
        >>> search_engine = create_search_engine_from_scratch()
        >>> search_engine.display_results("How to use datasets?", k=5)
    """
    from data_processor import DataProcessor
    
    print("=" * 80)
    print("CREATING SEMANTIC SEARCH ENGINE FROM SCRATCH")
    print("=" * 80 + "\n")
    
    # Step 1: Process data
    print("STEP 1: Processing data...")
    processor = DataProcessor(dataset_name)
    dataset = processor.preprocess_full_pipeline(
        split='train',
        min_comment_length=min_comment_length
    )
    
    # Step 2: Generate embeddings
    print("\nSTEP 2: Generating embeddings...")
    generator = EmbeddingGenerator(model_checkpoint)
    dataset_with_embeddings = generator.generate_embeddings_for_dataset(
        dataset,
        batch_size=batch_size
    )
    
    # Step 3: Build search engine
    print("\nSTEP 3: Building search engine...")
    search_engine = SemanticSearchEngine(generator)
    search_engine.build_index(dataset_with_embeddings)
    
    print("\n" + "=" * 80)
    print("SEMANTIC SEARCH ENGINE READY!")
    print("=" * 80 + "\n")
    
    return search_engine
