
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


print("  HPE ONTOLOGY PROJECT — Cricket Domain")
print(" Taxonomy Building ")

print("\n" + "─" * 70)
print("SECTION 1 — DATASET (25 sentences)")

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



# TAXONOMY BUILDING


print("\n" + "═" * 70)
print("  STEP 4 — TAXONOMY BUILDING")



# METHOD - HEARST PATTERN-BASED

print("\n" + "─" * 70)
print("METHOD 1 — Hearst Pattern-Based Taxonomy")

hearst_patterns = [
    # (child, parent) direction
    (r"([A-Za-z ]+) is a (?:type of |kind of |form of )?([A-Za-z ]+)", "is-a"),
    (r"([A-Za-z ]+) is an ([A-Za-z ]+)", "is-a"),
    (r"(?:types of|formats of|kinds of) ([A-Za-z]+) include ([A-Za-z ,and]+)", "hyponymy"),
    (r"([A-Za-z ,and]+) (?:such as|including) ([A-Za-z ,and]+)", "such-as"),
]

hearst_pairs = []

for sent in sentences:
    # Pattern 1: "X is a Y" / "X is an Y"
    m = re.search(r"^([A-Za-z\s]+) is an? (?:type of |kind of |form of )?([A-Za-z\s]+?)(?:who|that|which|$)", sent, re.I)
    if m:
        child = m.group(1).strip().lower()
        parent = m.group(2).strip().lower()
        # filter very long/noisy parents
        if 1 <= len(parent.split()) <= 4 and 1 <= len(child.split()) <= 4:
            hearst_pairs.append((child, parent, "is-a", sent))

    # Pattern 2: "types/formats of X include A, B, C"
    m2 = re.search(r"(?:types|formats|kinds) of ([A-Za-z]+) include ([A-Za-z ,and]+)", sent, re.I)
    if m2:
        parent_term = m2.group(1).strip().lower()
        children_raw = m2.group(2).strip().lower()
        children = [c.strip().rstrip(".") for c in re.split(r",| and ", children_raw) if c.strip()]
        for ch in children:
            if ch:
                hearst_pairs.append((ch, parent_term, "hyponym-of", sent))

    # Pattern 3: "X such as A, B"
    m3 = re.search(r"([A-Za-z\s]+) (?:such as|including) ([A-Za-z ,and]+)", sent, re.I)
    if m3:
        parent_term = m3.group(1).strip().lower().split()[-1]  # last word as parent
        children_raw = m3.group(2).strip().lower()
        children = [c.strip().rstrip(".") for c in re.split(r",| and ", children_raw) if c.strip()]
        for ch in children:
            if ch:
                hearst_pairs.append((ch, parent_term, "such-as", sent))

# Deduplicate
seen = set()
unique_hearst = []
for row in hearst_pairs:
    key = (row[0], row[1])
    if key not in seen and row[0] != row[1]:
        seen.add(key)
        unique_hearst.append(row)

# Build taxonomy dict
taxonomy_hearst = defaultdict(list)
for child, parent, rel, _ in unique_hearst:
    taxonomy_hearst[parent].append(child)

print("\n  HEARST TAXONOMY PAIRS (child → parent):")
df_hearst = pd.DataFrame([(r[0], r[1], r[2]) for r in unique_hearst],
                         columns=["Child", "Parent", "Relation"])
print(df_hearst.to_string(index=False))

print("\n  HIERARCHICAL STRUCTURE (parent → children):")
for parent in sorted(taxonomy_hearst):
    children = sorted(set(taxonomy_hearst[parent]))
    print(f"    {parent.upper()}")
    for ch in children:
        print(f"      └── {ch}")

# ─────────────────────────────────────────────
# VISUALIZATION — TREE STRUCTURE
# ─────────────────────────────────────────────

print("\n" + "─" * 70)
print("VISUALIZATION — Tree Structure")

import matplotlib.pyplot as plt
import networkx as nx

# Create graph
G = nx.DiGraph()

# Add edges
for parent, children in taxonomy_hearst.items():
    for child in children:
        G.add_edge(parent, child)

# Add a ROOT node to connect everything
root = "cricket"
G.add_node(root)

for parent in taxonomy_hearst:
    G.add_edge(root, parent)

# Function to create hierarchical layout
def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):

    def _hierarchy_pos(G, node, width, vert_gap, vert_loc, xcenter, pos, visited):
        if node in visited:
            return pos
        visited.add(node)

        children = list(G.successors(node))
        if not children:
            return pos

        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2

        for child in children:
            nextx += dx
            if child not in pos:
                pos[child] = (nextx, vert_loc - vert_gap)
            pos = _hierarchy_pos(G, child, dx, vert_gap, vert_loc - vert_gap, nextx, pos, visited)

        return pos

    pos = {root: (xcenter, vert_loc)}
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter, pos, set())

# Generate positions
pos = hierarchy_pos(G, root)

# Draw
plt.figure(figsize=(12, 8))
nx.draw(
    G, pos,
    with_labels=True,
    node_size=2500,
    node_color="lightgreen",
    font_size=9,
    font_weight="bold",
    arrows=True
)

plt.title("Cricket Ontology — Tree View")

plt.show()