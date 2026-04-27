1. Overview
This project implements a system to construct a taxonomy (hierarchical structure) from cricket-related sentences.
The program processes input sentences and extracts relationships such as:
batsman → player
spinner → bowler
These relationships are then organized into a structured hierarchy.

2. Dataset
A dataset consisting of 25 cricket-related sentences was manually created and used as input for the system.

3. Implementation
3.1 Preprocessing
Converted all text to lowercase
Removed punctuation

3.2 Keyword Extraction
Removed common stopwords (e.g., “is”, “the”)
Retained meaningful keywords

3.3 Pattern Matching
Pattern-based methods were used to extract relationships from text, including:
“X is a Y”
“types of X include Y”
“X such as Y”
These patterns were used to identify and extract entity relationships.

4. Taxonomy Construction
Extracted relationships were stored in the form:
child → parent
These were then reorganized into a hierarchical structure:
parent → children

5. Technologies Used
Language: Python
Libraries: pandas, numpy, re, networkx, matplotlib, scikit-learn

6. Visualization
The constructed taxonomy was visualized using NetworkX and Matplotlib.
Entities are represented as nodes, and relationships between them are represented as edges in a graph.
