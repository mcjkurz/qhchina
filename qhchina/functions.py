# qhchina/functions.py

import subprocess
import sys
import importlib

def tokenize(text, model=None):
    # Check if spacy is already installed
    spacy = importlib.util.find_spec("spacy")
    if spacy is None:
        # Install spacy if not installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    
    if model == "zh_core_web_lg":
        # Download the Chinese model using subprocess
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "zh_core_web_lg"])
    
    # Import spacy after installing the model
    import spacy
    
    # Load the Chinese model
    nlp = spacy.load("zh_core_web_lg")
    doc = nlp(text)
    return [token.text for token in doc]

def hello():
    """Prints welcome message."""
    print("Hello!")