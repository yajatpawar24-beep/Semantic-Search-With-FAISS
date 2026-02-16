"""Setup script for Semantic Search with FAISS package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="semantic-search-faiss",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Semantic search engine using sentence transformers and FAISS for efficient similarity search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/semantic-search-faiss",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "semantic-search-demo=demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
