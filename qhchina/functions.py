# qhchina/functions.py

def tokenize(text, model=None):
    if model == "zh_core_web_lg":
        !python3 -m spacy download zh_core_web_lg
    import spacy
    nlp = spacy.load("zh_core_web_lg")
    doc = nlp(text)
    return [token.text for token in doc]

def hello():
    """Prints welcome message."""
    print("Hello!")