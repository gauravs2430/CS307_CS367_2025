import heapq
import re

def preprocess_text(text):
    """
    Normalize text by:
    - Lowercasing
    - Removing punctuation
    - Splitting into sentences
    """
    text = text.lower()
    text = re.sub(r'[^\w\s.]', '', text)   # remove punctuation except '.'
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sentences


# levenshtein Edit Distance
def edit_distance(s1, s2):
    
    # accourding to the algorithm given in book 
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[m][n]


# A* Search for Alignment
def a_star_alignment(doc1_sentences, doc2_sentences, skip_penalty=5):
    
    start_state = (0, 0)  # (i, j)
    goal_state = (len(doc1_sentences), len(doc2_sentences))

    # Priority Queue: (f(n), g(n), state, path)
    frontier = []
    heapq.heappush(frontier, (0, 0, start_state, []))
    visited = set()

    while frontier:
        f, g, (i, j), path = heapq.heappop(frontier)

        
        if (i, j) == goal_state:
            return path, g

        if (i, j) in visited:
            continue
        visited.add((i, j))

        # Option 1: Match both sentences
        if i < len(doc1_sentences) and j < len(doc2_sentences):
            cost = edit_distance(doc1_sentences[i], doc2_sentences[j])
            new_state = (i+1, j+1)
            new_g = g + cost
            new_h = (len(doc1_sentences)-i-1) + (len(doc2_sentences)-j-1)
            new_f = new_g + new_h
            heapq.heappush(frontier, (new_f, new_g, new_state, path + [("MATCH", doc1_sentences[i], doc2_sentences[j], cost)]))

        # Option 2: Skip sentence in Doc1
        if i < len(doc1_sentences):
            new_state = (i+1, j)
            new_g = g + skip_penalty
            new_h = (len(doc1_sentences)-i-1) + (len(doc2_sentences)-j)
            new_f = new_g + new_h
            heapq.heappush(frontier, (new_f, new_g, new_state, path + [("SKIP_DOC1", doc1_sentences[i], None, skip_penalty)]))

        # Option 3: Skip sentence in Doc2
        if j < len(doc2_sentences):
            new_state = (i, j+1)
            new_g = g + skip_penalty
            new_h = (len(doc1_sentences)-i) + (len(doc2_sentences)-j-1)
            new_f = new_g + new_h
            heapq.heappush(frontier, (new_f, new_g, new_state, path + [("SKIP_DOC2", None, doc2_sentences[j], skip_penalty)]))

    return None, float("inf")  # No alignment found



# 4. Detect Plagiarism
def detect_plagiarism(path, threshold=3):

    plagiarism_pairs = []
    for action, s1, s2, cost in path:
        if action == "MATCH" and cost <= threshold:
            plagiarism_pairs.append((s1, s2, cost))
    return plagiarism_pairs



# 5. Main Function (Test)
if __name__ == "__main__":
    doc1 = "AI helps in many fields. Machine learning is part of AI. It makes computers smart."
    doc2 = "AI helps in many areas. It makes computers intelligent."

    doc1_sentences = preprocess_text(doc1)
    doc2_sentences = preprocess_text(doc2)

    path, total_cost = a_star_alignment(doc1_sentences, doc2_sentences)

    print("\nAlignment Path:")
    for step in path:
        print(step)

    print("\nTotal Cost:", total_cost)

    plagiarism = detect_plagiarism(path)
    print("\nPotential Plagiarism Pairs (low edit distance):")
    for pair in plagiarism:
        print(pair)
