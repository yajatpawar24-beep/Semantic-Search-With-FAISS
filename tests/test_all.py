"""
Comprehensive Test Suite for Semantic Search with FAISS

This module contains unit tests and integration tests for all components
of the semantic search system.
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from datasets import Dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processor import DataProcessor, get_dataset_sample
from embeddings import EmbeddingGenerator, compare_embeddings_similarity
from semantic_search import SemanticSearchEngine


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        print("\n" + "="*70)
        print("Testing Data Processing Module")
        print("="*70)
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor('lewtun/github-issues')
        self.assertEqual(processor.dataset_name, 'lewtun/github-issues')
        self.assertIsNone(processor.dataset)
    
    def test_custom_dataset_name(self):
        """Test initialization with custom dataset name."""
        custom_name = 'custom/dataset'
        processor = DataProcessor(custom_name)
        self.assertEqual(processor.dataset_name, custom_name)
    
    def test_filter_pull_requests(self):
        """Test filtering pull requests and empty comments."""
        # Create a small mock dataset
        data = {
            'is_pull_request': [False, True, False, False],
            'comments': [['comment1'], [], ['comment2'], ['comment3']],
            'title': ['Issue 1', 'PR 1', 'Issue 2', 'Issue 3'],
            'body': ['Body 1', 'Body 2', 'Body 3', 'Body 4'],
            'html_url': ['url1', 'url2', 'url3', 'url4']
        }
        mock_dataset = Dataset.from_dict(data)
        
        processor = DataProcessor()
        filtered = processor.filter_pull_requests_and_empty_comments(mock_dataset)
        
        # Should keep only non-PR entries with comments
        self.assertEqual(len(filtered), 3)
    
    def test_remove_unnecessary_columns(self):
        """Test removing unnecessary columns."""
        data = {
            'title': ['Title 1', 'Title 2'],
            'body': ['Body 1', 'Body 2'],
            'html_url': ['url1', 'url2'],
            'comments': [['c1'], ['c2']],
            'extra_column': ['extra1', 'extra2'],
            'another_column': ['another1', 'another2']
        }
        mock_dataset = Dataset.from_dict(data)
        
        processor = DataProcessor()
        cleaned = processor.remove_unnecessary_columns(
            mock_dataset,
            columns_to_keep=['title', 'body', 'html_url', 'comments']
        )
        
        # Check that only specified columns remain
        self.assertIn('title', cleaned.column_names)
        self.assertIn('comments', cleaned.column_names)
        self.assertNotIn('extra_column', cleaned.column_names)
    
    def test_explode_comments(self):
        """Test exploding comments to individual rows."""
        data = {
            'title': ['Issue 1', 'Issue 2'],
            'body': ['Body 1', 'Body 2'],
            'html_url': ['url1', 'url2'],
            'comments': [['comment1', 'comment2'], ['comment3']]
        }
        mock_dataset = Dataset.from_dict(data)
        
        processor = DataProcessor()
        exploded = processor.explode_comments(mock_dataset)
        
        # Should have 3 rows (2 comments from first issue + 1 from second)
        self.assertEqual(len(exploded), 3)
    
    def test_filter_short_comments(self):
        """Test filtering comments that are too short."""
        data = {
            'title': ['Issue 1', 'Issue 2', 'Issue 3'],
            'body': ['Body 1', 'Body 2', 'Body 3'],
            'html_url': ['url1', 'url2', 'url3'],
            'comments': [
                'This is a long comment with more than fifteen words in it for testing',
                'Short',
                'This comment has exactly sixteen words in it so it should pass the filter test'
            ]
        }
        mock_dataset = Dataset.from_dict(data)
        
        processor = DataProcessor()
        filtered = processor.filter_short_comments(mock_dataset, min_word_count=15)
        
        # Should keep only comments with > 15 words
        self.assertGreater(len(mock_dataset), len(filtered))
    
    def test_concatenate_text_fields(self):
        """Test concatenating title, body, and comments."""
        data = {
            'title': ['Title 1'],
            'body': ['Body 1'],
            'comments': ['Comment 1']
        }
        mock_dataset = Dataset.from_dict(data)
        
        processor = DataProcessor()
        with_text = processor.concatenate_text_fields(mock_dataset)
        
        # Check that text column was added
        self.assertIn('text', with_text.column_names)
        
        # Check that text contains all fields
        text = with_text[0]['text']
        self.assertIn('Title 1', text)
        self.assertIn('Body 1', text)
        self.assertIn('Comment 1', text)
    
    def test_get_dataset_sample(self):
        """Test getting dataset sample as DataFrame."""
        data = {
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        }
        mock_dataset = Dataset.from_dict(data)
        
        sample = get_dataset_sample(mock_dataset, n=3)
        
        self.assertIsInstance(sample, pd.DataFrame)
        self.assertEqual(len(sample), 3)


class TestEmbeddingGenerator(unittest.TestCase):
    """Test cases for EmbeddingGenerator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        print("\n" + "="*70)
        print("Testing Embedding Generation Module")
        print("="*70)
        cls.generator = EmbeddingGenerator(
            "sentence-transformers/all-MiniLM-L6-v2"  # Smaller model for faster tests
        )
    
    def test_initialization(self):
        """Test EmbeddingGenerator initialization."""
        self.assertIsNotNone(self.generator.model)
        self.assertIsNotNone(self.generator.tokenizer)
        self.assertIn(str(self.generator.device), ['cpu', 'cuda'])
    
    def test_single_text_embedding(self):
        """Test generating embedding for a single text."""
        text = "This is a test sentence."
        embedding = self.generator.get_embeddings(text)
        
        # Check shape
        self.assertEqual(len(embedding.shape), 2)
        self.assertEqual(embedding.shape[0], 1)  # 1 text
        
        # Check type
        self.assertIsInstance(embedding, np.ndarray)
    
    def test_multiple_text_embeddings(self):
        """Test generating embeddings for multiple texts."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = self.generator.get_embeddings(texts)
        
        # Check shape
        self.assertEqual(embeddings.shape[0], 3)  # 3 texts
        
        # Check that embeddings are different
        self.assertFalse(np.allclose(embeddings[0], embeddings[1]))
    
    def test_embedding_dimension(self):
        """Test getting embedding dimension."""
        dim = self.generator.get_embedding_dimension()
        
        self.assertIsInstance(dim, int)
        self.assertGreater(dim, 0)
    
    def test_embed_single_text_method(self):
        """Test the embed_single_text convenience method."""
        text = "Test query"
        embedding = self.generator.embed_single_text(text)
        
        # Should return array with shape (1, dim)
        self.assertEqual(len(embedding.shape), 2)
        self.assertEqual(embedding.shape[0], 1)
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        # Should not crash with empty string
        embedding = self.generator.get_embeddings([""])
        self.assertIsNotNone(embedding)
    
    def test_similarity_computation(self):
        """Test embedding similarity computation."""
        emb1 = self.generator.get_embeddings(["cat"])[0]
        emb2 = self.generator.get_embeddings(["dog"])[0]
        emb3 = self.generator.get_embeddings(["automobile"])[0]
        
        # Cat and dog should be more similar than cat and automobile
        similarity_cat_dog = compare_embeddings_similarity(emb1, emb2, metric='cosine')
        similarity_cat_auto = compare_embeddings_similarity(emb1, emb3, metric='cosine')
        
        self.assertGreater(similarity_cat_dog, similarity_cat_auto)


class TestSemanticSearch(unittest.TestCase):
    """Test cases for SemanticSearchEngine class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with a small dataset."""
        print("\n" + "="*70)
        print("Testing Semantic Search Module")
        print("="*70)
        
        # Create a small test dataset
        data = {
            'title': [
                'How to load datasets',
                'Training neural networks',
                'Data preprocessing tips',
                'Model evaluation metrics',
                'Loading pretrained models'
            ],
            'body': [
                'I need help loading datasets from disk',
                'What is the best way to train a neural network',
                'How should I preprocess my data',
                'Which metrics should I use for evaluation',
                'How do I load a pretrained model'
            ],
            'comments': [
                'You can use load_from_disk method',
                'Use gradient descent with proper learning rate',
                'Normalize your data first',
                'Use accuracy for classification tasks',
                'Use from_pretrained method'
            ],
            'html_url': [
                'http://example.com/1',
                'http://example.com/2',
                'http://example.com/3',
                'http://example.com/4',
                'http://example.com/5'
            ]
        }
        
        # Add concatenated text
        cls.test_dataset = Dataset.from_dict(data)
        cls.test_dataset = cls.test_dataset.map(
            lambda x: {
                'text': x['title'] + ' ' + x['body'] + ' ' + x['comments']
            }
        )
        
        # Generate embeddings
        cls.generator = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")
        
        embeddings_list = []
        for text in cls.test_dataset['text']:
            emb = cls.generator.get_embeddings([text])[0]
            embeddings_list.append(emb.tolist())
        
        cls.test_dataset = cls.test_dataset.add_column('embeddings', embeddings_list)
        
        # Create search engine
        cls.search_engine = SemanticSearchEngine(cls.generator)
        cls.search_engine.build_index(cls.test_dataset)
    
    def test_search_engine_initialization(self):
        """Test search engine initialization."""
        engine = SemanticSearchEngine()
        self.assertIsNotNone(engine.embedding_generator)
    
    def test_index_building(self):
        """Test FAISS index building."""
        self.assertTrue(self.search_engine.index_built)
        self.assertIsNotNone(self.search_engine.dataset)
    
    def test_basic_search(self):
        """Test basic search functionality."""
        query = "How to load data"
        scores, samples = self.search_engine.search(query, k=3)
        
        # Check that we got results
        self.assertEqual(len(scores), 3)
        self.assertIn('title', samples)
        
        # Check that scores are in descending order
        self.assertGreaterEqual(scores[0], scores[1])
        self.assertGreaterEqual(scores[1], scores[2])
    
    def test_search_and_format(self):
        """Test search with DataFrame formatting."""
        query = "training models"
        results_df = self.search_engine.search_and_format(query, k=2)
        
        # Check DataFrame properties
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), 2)
        self.assertIn('scores', results_df.columns)
        self.assertIn('title', results_df.columns)
    
    def test_search_without_index(self):
        """Test that search fails without building index."""
        engine = SemanticSearchEngine(self.generator)
        
        with self.assertRaises(RuntimeError):
            engine.search("test query")
    
    def test_search_relevance(self):
        """Test that search returns relevant results."""
        query = "How to load datasets"
        results_df = self.search_engine.search_and_format(query, k=5)
        
        # The top result should be about loading datasets
        top_title = results_df.iloc[0]['title'].lower()
        self.assertIn('load', top_title)


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end workflows."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        print("\n" + "="*70)
        print("Running Integration Tests")
        print("="*70)
    
    def test_full_pipeline_small_dataset(self):
        """Test complete pipeline with a tiny dataset."""
        # Create minimal dataset
        data = {
            'is_pull_request': [False, False],
            'title': ['Test 1', 'Test 2'],
            'body': ['Body 1', 'Body 2'],
            'comments': [['Long comment with more than fifteen words to pass the filter test'], 
                        ['Another long comment with more than fifteen words for testing purposes']],
            'html_url': ['url1', 'url2']
        }
        dataset = Dataset.from_dict(data)
        
        # Process data
        processor = DataProcessor()
        dataset = processor.filter_pull_requests_and_empty_comments(dataset)
        dataset = processor.remove_unnecessary_columns(dataset)
        dataset = processor.concatenate_text_fields(dataset)
        
        # Generate embeddings
        generator = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")
        
        embeddings_list = []
        for text in dataset['text']:
            emb = generator.get_embeddings([text])[0]
            embeddings_list.append(emb.tolist())
        
        dataset = dataset.add_column('embeddings', embeddings_list)
        
        # Build search engine
        search_engine = SemanticSearchEngine(generator)
        search_engine.build_index(dataset)
        
        # Perform search
        results = search_engine.search_and_format("test query", k=1)
        
        # Verify results
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)
    
    def test_batch_search_workflow(self):
        """Test batch search functionality."""
        # Create small dataset
        data = {
            'title': ['Python help', 'JavaScript help', 'Java help'],
            'body': ['Python question', 'JavaScript question', 'Java question'],
            'comments': ['Python answer here', 'JavaScript answer here', 'Java answer here'],
            'html_url': ['url1', 'url2', 'url3']
        }
        dataset = Dataset.from_dict(data)
        dataset = dataset.map(lambda x: {'text': x['title'] + ' ' + x['body'] + ' ' + x['comments']})
        
        # Generate embeddings
        generator = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")
        embeddings_list = []
        for text in dataset['text']:
            emb = generator.get_embeddings([text])[0]
            embeddings_list.append(emb.tolist())
        dataset = dataset.add_column('embeddings', embeddings_list)
        
        # Build search engine
        search_engine = SemanticSearchEngine(generator)
        search_engine.build_index(dataset)
        
        # Batch search
        queries = ['Python', 'JavaScript']
        results = search_engine.batch_search(queries, k=2)
        
        # Verify
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], pd.DataFrame)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        print("\n" + "="*70)
        print("Testing Edge Cases and Error Handling")
        print("="*70)
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        data = {'col1': [], 'col2': []}
        empty_dataset = Dataset.from_dict(data)
        
        processor = DataProcessor()
        # Should not crash
        result = processor.remove_unnecessary_columns(
            empty_dataset,
            columns_to_keep=['col1']
        )
        self.assertEqual(len(result), 0)
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        generator = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")
        
        # Create very long text (will be truncated)
        long_text = "word " * 1000
        embedding = generator.get_embeddings([long_text])
        
        # Should still produce valid embedding
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding.shape), 2)
    
    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        generator = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")
        
        special_text = "Test with émojis 😀 and spëcial çhars!"
        embedding = generator.get_embeddings([special_text])
        
        # Should handle gracefully
        self.assertIsNotNone(embedding)
    
    def test_search_with_k_larger_than_dataset(self):
        """Test search when k is larger than dataset size."""
        # Create small dataset
        data = {
            'title': ['Item 1', 'Item 2'],
            'text': ['Text 1', 'Text 2'],
            'html_url': ['url1', 'url2']
        }
        dataset = Dataset.from_dict(data)
        
        generator = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")
        embeddings_list = []
        for text in dataset['text']:
            emb = generator.get_embeddings([text])[0]
            embeddings_list.append(emb.tolist())
        dataset = dataset.add_column('embeddings', embeddings_list)
        
        search_engine = SemanticSearchEngine(generator)
        search_engine.build_index(dataset)
        
        # Request more results than available
        scores, samples = search_engine.search("query", k=10)
        
        # Should return only available results
        self.assertLessEqual(len(scores), 2)


def run_tests(verbosity=2):
    """
    Run all tests.
    
    Args:
        verbosity: Test output verbosity level (0-2)
    
    Returns:
        TestResult object
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticSearch))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("\n" + "="*80)
    print("SEMANTIC SEARCH WITH FAISS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    result = run_tests(verbosity=2)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
