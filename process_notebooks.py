"""
Script to edit Jupyter notebooks to present them as original work
rather than course exercises.
"""
import json
import os
import re
from pathlib import Path

def clean_markdown_cell(cell_source):
    """
    Clean markdown cell content to remove course/exercise language
    and present as original implementation.
    """
    text = ''.join(cell_source)
    
    # Remove specific course-related sections (must be done first)
    removals = [
        (r'Welcome to your week \d+ assignment[^\n]*\n+', ''),
        (r'\*\*By the end of this assignment.*?\n\n', ''),
        (r'## Important Note on Submission to the AutoGrader.*?(?=##[^#]|\Z)', ''),
        (r'Before submitting your assignment.*?(?=##[^#]|\Z)', ''),
        (r'## Table of Contents.*?(?=##[^#]|\Z)', ''),
        (r'<a name=[\'"]?[^>]*>[^<]*</a>\n?', ''),
        (r'### Exercise \d+ - ', '### '),
        (r'\*\*\*Expected output\*\*\*\n```.*?```\n?', ''),
        (r'#\(≈[^\)]*\)[^\n]*\n', ''),
        (r'# YOUR CODE STARTS HERE[^\n]*\n', ''),
        (r'# YOUR CODE ENDS HERE[^\n]*\n', ''),
    ]
    
    for pattern, replacement in removals:
        text = re.sub(pattern, replacement, text, flags=re.DOTALL | re.MULTILINE)
    
    # Word-level replacements (be careful with word boundaries)
    word_replacements = [
        (r'\bImplement\b', 'Implementation of'),
        (r'\bBuild\b', 'Building'),
        (r'\bCreate and initialize\b', 'Initialization of'),
    ]
    
    for pattern, replacement in word_replacements:
        text = re.sub(pattern, replacement, text)
    
    # Phrase replacements
    phrase_replacements = [
        ('you will implement', 'this implements'),
        ('you\'ll implement', 'this implements'),
        ('you will', 'this'),
        ('you\'ll', 'this will'),
        ('your assignment', 'this project'),
        ('this assignment', 'this implementation'),
        ('You have', 'This has'),
        ('you have', 'this has'),
        ('You\'ve been provided', 'Using'),
        ('you\'ve been provided', 'using'),
    ]
    
    for old, new in phrase_replacements:
        text = text.replace(old, new)
        text = text.replace(old.capitalize(), new.capitalize())
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return [line + '\n' for line in text.split('\n')] if text else []

def add_professional_header(notebook_name):
    """Generate a professional header for the notebook."""
    headers = {
        'Building_your_Deep_Neural_Network_Step_by_Step': {
            'title': '# Deep Neural Network Implementation',
            'description': '''
## Project Overview

Complete implementation of an L-layer deep neural network from scratch using NumPy. This project demonstrates fundamental understanding of neural network architecture, forward/backward propagation, and gradient descent optimization.

### Key Features
- Modular design supporting any number of layers
- ReLU and Sigmoid activation functions
- Efficient vectorized operations
- Forward and backward propagation
- Parameter initialization strategies

### Technical Implementation
- **Framework**: NumPy (built from scratch)
- **Architecture**: Configurable L-layer network
- **Optimization**: Gradient descent with backpropagation
'''
        },
        'Deep Neural Network - Application': {
            'title': '# Deep Neural Network - Image Classification',
            'description': '''
## Project Overview

Binary image classification application using a deep neural network. This project applies the L-layer neural network framework to real-world image data, demonstrating end-to-end machine learning workflow.

### Objectives
- Preprocess and normalize image datasets
- Train deep neural network for binary classification
- Analyze model performance and decision boundaries
- Optimize hyperparameters for best results
'''
        },
        'Planar_data_classification_with_one_hidden_layer': {
            'title': '# Planar Data Classification with Neural Network',
            'description': '''
## Project Overview

Neural network implementation for classifying non-linear planar data. Demonstrates the power of hidden layers in learning complex decision boundaries that linear models cannot capture.

### Key Achievements
- Single hidden layer neural network from scratch
- Non-linear decision boundary learning
- Visualization of model predictions
- Performance comparison with logistic regression baseline
'''
        },
        'Python_Basics_with_Numpy': {
            'title': '# NumPy Fundamentals for Deep Learning',
            'description': '''
## Project Overview

Foundation for efficient deep learning computations using NumPy. This notebook covers vectorization techniques and mathematical operations essential for neural network implementations.

### Core Concepts
- Vectorization for performance optimization
- Broadcasting and array manipulation
- Mathematical operations for neural networks
- Sigmoid function and gradient computation
'''
        }
    }
    
    # Default header if specific one not found
    default = {
        'title': f'# {notebook_name.replace("_", " ").replace(".ipynb", "")}',
        'description': '''
## Project Overview

Implementation of deep learning concepts and techniques.
'''
    }
    
    header = headers.get(notebook_name.replace('.ipynb', ''), default)
    return [header['title'] + '\n', header['description']]

def process_notebook(notebook_path):
    """Process a single notebook file."""
    print(f"Processing: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Add professional header as first cell if first cell is markdown
    if notebook['cells'] and notebook['cells'][0]['cell_type'] == 'markdown':
        # Replace first markdown cell with professional header
        notebook_name = Path(notebook_path).name
        new_header = add_professional_header(notebook_name)
        notebook['cells'][0]['source'] = new_header
    
    # Process all markdown cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            cell['source'] = clean_markdown_cell(cell['source'])
    
    # Save processed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Completed: {notebook_path}")

def main():
    """Process all notebooks in the repository."""
    base_path = Path(r'c:/Users/Genia/OneDrive/Escritorio/Deeplearning.ai')
    
    # Find all .ipynb files
    notebooks = list(base_path.rglob('*.ipynb'))
    
    print(f"Found {len(notebooks)} notebooks to process\n")
    
    for notebook_path in notebooks:
        try:
            process_notebook(notebook_path)
        except Exception as e:
            print(f"✗ Error processing {notebook_path}: {e}")
    
    print(f"\n✓ Processing complete! Processed {len(notebooks)} notebooks.")

if __name__ == '__main__':
    main()
