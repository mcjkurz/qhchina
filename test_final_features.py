#!/usr/bin/env python3
"""
Quick test to verify the final two features:
1. Title is centered relative to container, not screen
2. n_topic_words parameter controls number of words per topic
"""

from qhchina.analytics.topicmodels import LDAGibbsSampler
from qhchina.helpers import load_stopwords
from qhchina import load_fonts
import re
import random

print("Testing final features...")
print("=" * 60)

load_fonts("sans")

# Load stopwords and prepare data
stopwords = load_stopwords("zh_cl_tr")
numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
stopwords.update(numbers)

with open("tests/明史.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Prepare passages
passages = [p for p in text.split("\n") if len(p.strip()) > 150]
chinese_regex = re.compile(r'[\u4e00-\u9fff]')
passages = [[char for char in passage if char not in stopwords and chinese_regex.match(char)] 
            for passage in passages]

random.seed(42)
random.shuffle(passages)
passages = passages[:200]  # Use subset for faster testing

print(f"Using {len(passages)} passages for testing")

# Train the model
print("\nTraining LDA model...")
lda = LDAGibbsSampler(
    n_topics=10,
    iterations=30,
    burnin=5,
    log_interval=10,
    random_state=42
)

lda.fit(passages)

doc_labels = [f"Passage_{i}" for i in range(len(passages))]

# Test 1: Default 4 words per topic
print("\n" + "=" * 60)
print("Test 1: Default (4 words per topic)")
print("=" * 60)
lda.visualize_documents(
    method='pca',
    doc_labels=doc_labels,
    format='html',
    highlight=[0, 2, 5],
    filename='tests/test_default_4words.html'
)
print("✓ Created: tests/test_default_4words.html")
print("  - Title should be centered within container")
print("  - Should show 4 words per topic in legend")

# Test 2: Custom 6 words per topic
print("\n" + "=" * 60)
print("Test 2: Custom (6 words per topic)")
print("=" * 60)
lda.visualize_documents(
    method='pca',
    doc_labels=doc_labels,
    format='html',
    highlight=[0, 2, 5],
    n_topic_words=6,
    filename='tests/test_custom_6words.html'
)
print("✓ Created: tests/test_custom_6words.html")
print("  - Title should be centered within container")
print("  - Should show 6 words per topic in legend")

# Test 3: Minimal 2 words per topic
print("\n" + "=" * 60)
print("Test 3: Minimal (2 words per topic)")
print("=" * 60)
lda.visualize_documents(
    method='pca',
    doc_labels=doc_labels,
    format='html',
    n_topic_words=2,
    filename='tests/test_minimal_2words.html'
)
print("✓ Created: tests/test_minimal_2words.html")
print("  - Title should be centered within container")
print("  - Should show 2 words per topic in legend")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
print("\nInstructions:")
print("1. Open the generated HTML files in your browser")
print("2. Verify the title stays centered when you zoom in/out")
print("3. Verify each file shows the correct number of words per topic")
print("4. Try clicking on topics and points to toggle highlighting")

