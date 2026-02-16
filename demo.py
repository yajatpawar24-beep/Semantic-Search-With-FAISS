#!/usr/bin/env python3
"""
Interactive Demo for Semantic Search with FAISS

This script provides an interactive command-line interface for semantic search
over GitHub issues dataset.
"""

import argparse
import sys
from semantic_search import SemanticSearchEngine, create_search_engine_from_scratch


def run_interactive_demo(search_engine: SemanticSearchEngine, k: int = 5):
    """
    Run interactive search demo where users can input queries.
    
    Args:
        search_engine: Initialized SemanticSearchEngine instance
        k: Number of results to show per query
    """
    print("\n" + "=" * 80)
    print("SEMANTIC SEARCH INTERACTIVE DEMO")
    print("=" * 80)
    print(f"\nSearching over {len(search_engine.dataset)} GitHub issue comments")
    print(f"Showing top {k} results per query\n")
    print("Commands:")
    print("  - Type your question to search")
    print("  - Type 'k=N' to change number of results (e.g., 'k=10')")
    print("  - Type 'quit' or 'exit' to end")
    print("=" * 80 + "\n")
    
    current_k = k
    
    while True:
        try:
            # Get user input
            query = input("Enter your search query: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Semantic Search! Goodbye!")
                break
            
            # Check for empty query
            if not query:
                print("Please enter a non-empty query.\n")
                continue
            
            # Check for k parameter change
            if query.lower().startswith('k='):
                try:
                    current_k = int(query.split('=')[1])
                    print(f"Number of results changed to {current_k}\n")
                    continue
                except (ValueError, IndexError):
                    print("Invalid k value. Use format: k=5\n")
                    continue
            
            # Perform search
            search_engine.display_results(query, k=current_k, show_full_text=False)
            
        except KeyboardInterrupt:
            print("\n\nSearch interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError during search: {e}")
            print("Please try again.\n")


def run_example_queries(search_engine: SemanticSearchEngine, k: int = 5):
    """
    Run predefined example queries to demonstrate the system.
    
    Args:
        search_engine: Initialized SemanticSearchEngine instance
        k: Number of results to show per query
    """
    example_queries = [
        "How can I load a dataset offline?",
        "What's the best way to tokenize text?",
        "How to fine-tune a pretrained model?",
        "How to save and load model checkpoints?",
        "How to handle out of memory errors?"
    ]
    
    print("\n" + "=" * 80)
    print("RUNNING EXAMPLE QUERIES")
    print("=" * 80 + "\n")
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}/{len(example_queries)}")
        print(f"{'='*80}")
        search_engine.display_results(query, k=k, show_full_text=False)
        
        if i < len(example_queries):
            input("\nPress Enter to see next example...")


def run_batch_demo(search_engine: SemanticSearchEngine, k: int = 3):
    """
    Demonstrate batch search functionality.
    
    Args:
        search_engine: Initialized SemanticSearchEngine instance
        k: Number of results to show per query
    """
    queries = [
        "How to use the Trainer API?",
        "What are the best practices for data preprocessing?",
        "How to optimize model performance?"
    ]
    
    print("\n" + "=" * 80)
    print("BATCH SEARCH DEMO")
    print("=" * 80 + "\n")
    
    results = search_engine.batch_search(queries, k=k)
    
    for i, (query, result_df) in enumerate(zip(queries, results), 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*80}\n")
        print(result_df[['title', 'scores', 'html_url']].to_string(index=False))
        print()


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(
        description="Interactive Semantic Search Demo using FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive demo with default settings
  python demo.py --mode interactive
  
  # Run example queries
  python demo.py --mode examples --k 10
  
  # Run batch search demo
  python demo.py --mode batch
  
  # Use custom model
  python demo.py --model sentence-transformers/all-MiniLM-L6-v2
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'examples', 'batch'],
        default='interactive',
        help='Demo mode to run (default: interactive)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of results to return per query (default: 5)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/multi-qa-mpnet-base-dot-v1',
        help='Sentence transformer model to use'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='lewtun/github-issues',
        help='HuggingFace dataset to search (default: lewtun/github-issues)'
    )
    
    parser.add_argument(
        '--min-comment-length',
        type=int,
        default=15,
        help='Minimum word count for comments (default: 15)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Use a small subset for quick testing'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("SEMANTIC SEARCH WITH FAISS - DEMO")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Mode: {args.mode}")
    print(f"  Results per query: {args.k}")
    print()
    
    try:
        # Create search engine
        print("Initializing search engine...")
        print("This may take a few minutes on first run...\n")
        
        search_engine = create_search_engine_from_scratch(
            dataset_name=args.dataset,
            model_checkpoint=args.model,
            min_comment_length=args.min_comment_length,
            batch_size=args.batch_size
        )
        
        # For quick testing, limit dataset size
        if args.quick_test:
            print("\nQUICK TEST MODE: Using subset of data")
            search_engine.dataset = search_engine.dataset.select(range(1000))
            search_engine.build_index(search_engine.dataset)
        
        # Run appropriate demo mode
        if args.mode == 'interactive':
            run_interactive_demo(search_engine, k=args.k)
        elif args.mode == 'examples':
            run_example_queries(search_engine, k=args.k)
        elif args.mode == 'batch':
            run_batch_demo(search_engine, k=args.k)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
