1. What I did

I created a small project that builds a taxonomy (hierarchy) from cricket-related sentences.

The program reads sentences and finds relationships like:

batsman → player
spinner → bowler

Then it organizes them into a hierarchy.

2. Dataset

I used a dataset of 25 cricket sentences written manually in the code.

3. What I implemented
1. Preprocessing
Converted text to lowercase
Removed punctuation
2. Keyword Extraction
Removed common words (like “is”, “the”)
Kept important words
3. Pattern Matching (Main Part)

I used simple patterns like:

“X is a Y”
“types of X include Y”
“X such as Y”

Using this, I extracted relationships.

4. Taxonomy Building
Stored relations as:
child → parent
Converted into hierarchy:
parent → children
4. Technologies Used
Language: Python
Libraries:
pandas, numpy, re, networkx, matplotlib, scikit-learn
5. Visualization
I visualized the taxonomy using NetworkX and Matplotlib.
The relationships are shown as a graph where nodes represent entities and edges represent connections between them.