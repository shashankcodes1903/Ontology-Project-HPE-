1. Objective
Group cricket-related terms into a taxonomy using semantic similarity.

2. Dataset
25 manually written cricket sentences
Covers roles, formats, players, organizations

3. Method Used
Clustering-based taxonomy (unsupervised)

4. Implementation (Brief)
Preprocess text (lowercase, clean, tokenize)
Collect domain terms and entities
Build context for each term
Convert to vectors using TF-IDF
Compute similarity (cosine)
Apply Agglomerative Clustering
Assign labels using simple rules

5. Tools Used
Language:-Python
Libraries:-scikit-learn, NetworkX, Matplotlib, Pandas, NumPy

6. Output
Clustered taxonomy (text)
Graph visualization

7. Key Note
Produces semantic groups, not strict hierarchy (no “is-a” relations).