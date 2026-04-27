import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
#DATASET (25 sentences, Cricket)

sentences = [
    "Cricket is a sport played between two teams.",
    "A batsman is a player who bats in cricket.",
    "A bowler is a player who delivers the ball.",
    "A wicketkeeper is a fielder who stands behind the stumps.",
    "Test cricket is a format that includes matches lasting up to five days.",
    "ODI, such as One Day Internationals, is a limited-overs format.",
    "T20 is a format including matches of twenty overs per side.",
    "Formats of cricket include Test, ODI, and T20.",
    "Virat Kohli is a batsman who plays for India.",
    "Rohit Sharma is a batsman who captains India.",
    "Jasprit Bumrah is a bowler who plays for India.",
    "MS Dhoni is a wicketkeeper who led India to World Cup victory.",
    "Sachin Tendulkar is a batsman known as the God of Cricket.",
    "Shane Warne is a bowler who played for Australia.",
    "Ben Stokes is an all-rounder who plays for England.",
    "An all-rounder is a player who can both bat and bowl.",
    "A spinner is a type of bowler who uses spin to deceive batsmen.",
    "A pacer is a type of bowler who relies on pace and swing.",
    "Types of bowlers include spinners and pacers.",
    "The ICC is an organization that governs international cricket.",
    "The BCCI is an organization that controls cricket in India.",
    "The World Cup is a tournament organized by the ICC.",
    "The IPL is a tournament played in India featuring T20 cricket.",
    "A six is a shot where the batsman hits the ball beyond the boundary.",
    "A wicket is a target consisting of three stumps and two bails.",
]



print("Taxonomy Building")

print("\n" + "─" * 70)
print(" DATASET (25 sentences)")
for i, s in enumerate(sentences, 1):
    print(f"  S{i:02d}: {s}")

# PREPROCESSING

def preprocess(text):
    """Lowercase, strip punctuation for term extraction."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return text

def simple_tokenize(text):
    """Simple whitespace tokenizer (no NLTK dependency)."""
    return text.lower().split()

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "who", "that",
    "which", "with", "by", "as", "it", "its", "this", "he", "she", "they",
    "can", "could", "would", "will", "do", "does", "did", "has", "have", "had",
    "not", "no", "so", "up", "out", "where", "when", "how", "what", "both",
    "between", "beyond", "known", "uses", "relies", "stands", "behind"
}

def get_keywords(text):
    tokens = simple_tokenize(preprocess(text))
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

print("\n" + "─" * 70)
print("PREPROCESSING — Sample keyword extraction")
print("─" * 70)
for s in sentences[:4]:
    print(f"  TEXT : {s}")
    print(f"  KEYS : {get_keywords(s)}")
    print()



#TAXONOMY BUILDING

#CLUSTERING-BASED TAXONOMY


print("\n" + "─" * 70)
print("TF-IDF + Clustering-Based Taxonomy")


# Collect candidate terms (simple regex)
all_terms = set()
np_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s[A-Za-z]+)?)\b')
single_nouns = [
    "cricket", "batsman", "bowler", "wicketkeeper", "all-rounder",
    "spinner", "pacer", "fielder", "player", "format", "tournament",
    "organization", "sport", "match", "team", "boundary", "wicket",
    "stumps", "bails", "six", "shot", "delivery", "over"
]
for term in single_nouns:
    all_terms.add(term)

# Named entities
named = ["virat kohli", "rohit sharma", "jasprit bumrah", "ms dhoni",
         "sachin tendulkar", "shane warne", "ben stokes",
         "icc", "bcci", "ipl", "world cup", "india", "australia", "england"]
for n in named:
    all_terms.add(n)

terms_list = sorted(all_terms)

# Building a "context sentence" for each term
def get_context(term):
    ctx = []
    for s in sentences:
        if term.lower() in s.lower():
            ctx.append(s)
    return " ".join(ctx) if ctx else term

term_contexts = [get_context(t) for t in terms_list]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
X = vectorizer.fit_transform(term_contexts)

# Cosine similarity
sim_matrix = cosine_similarity(X)

# Agglomerative clustering
n_clusters = 6
clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed',
                                      linkage='complete')
distance_matrix = 1 - np.clip(sim_matrix, 0, 1)
labels = clustering.fit_predict(distance_matrix)

# Heuristic cluster labels (based on dominant terms in each cluster)
cluster_label_map = {
    0: "Player Roles",
    1: "Cricket Formats",
    2: "Notable Players",
    3: "Governing Bodies",
    4: "Game Elements",
    5: "Teams & Nations",
}

# Assign heuristic labels by inspecting cluster content
cluster_contents = defaultdict(list)
for term, label in zip(terms_list, labels):
    cluster_contents[label].append(term)

# Reassign human-readable labels heuristically
def heuristic_label(cluster_id, members):
    m = " ".join(members)
    if any(x in m for x in ["kohli", "sharma", "bumrah", "dhoni", "tendulkar", "warne", "stokes"]):
        return "Notable Players"
    if any(x in m for x in ["icc", "bcci", "organization"]):
        return "Governing Bodies"
    if any(x in m for x in ["test", "odi", "t20", "format", "ipl", "tournament"]):
        return "Cricket Formats"
    if any(x in m for x in ["batsman", "bowler", "spinner", "pacer", "all-rounder", "fielder", "wicketkeeper"]):
        return "Player Roles"
    if any(x in m for x in ["india", "australia", "england", "team"]):
        return "Teams & Nations"
    return "Game Elements"

cluster_named = {}
for cid, members in cluster_contents.items():
    cluster_named[cid] = heuristic_label(cid, members)

print("\n  CLUSTER ASSIGNMENTS:")
rows = []
for cid in sorted(cluster_contents.keys()):
    members = cluster_contents[cid]
    label = cluster_named[cid]
    for m in members:
        rows.append((m, label, f"Cluster {cid}"))

df_clusters = pd.DataFrame(rows, columns=["Term", "Cluster Label", "Cluster ID"])
print(df_clusters.to_string(index=False))

print("\n  CLUSTER HIERARCHY (Cluster → Members):")
for cid in sorted(cluster_contents.keys()):
    print(f"\n    [{cluster_named[cid]}]")
    for m in sorted(cluster_contents[cid]):
        print(f"      └── {m}")

print("\n" + "=" * 70)
print("CLUSTER VISUALIZATION (inline)")
print("=" * 70)

G = nx.Graph()

# Build graph
for cid, members in cluster_contents.items():
    cluster_name = cluster_named[cid]
    
    G.add_node(cluster_name)
    
    for term in members:
        G.add_node(term)
        G.add_edge(cluster_name, term)
pos = nx.spring_layout(G, seed=42, k=2.0)
node_colors = []
for node in G.nodes():
    if node in cluster_named.values():
        node_colors.append('orange')   # cluster
    else:
        node_colors.append('skyblue')  # term
plt.figure(figsize=(14, 10))
nx.draw(G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2000,
        font_size=9)
import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color='orange', label='Cluster'),
    mpatches.Patch(color='skyblue', label='Term')
]
plt.legend(handles=legend_elements)
plt.title("Clustering-Based Taxonomy (Term → Cluster)")
plt.axis('off')
plt.show()