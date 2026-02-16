# Research Plan: Multilingual Word Embedding Similarity and Polysemy

## Phase 0: Motivation & Novelty Assessment

### Why This Research Matters
Understanding how multilingual models represent concepts across languages is fundamental to cross-lingual NLP. If we know how "meaning" is preserved or distorted in the embedding space, we can better design models for low-resource languages and improve cross-lingual transfer. Specifically, understanding how polysemy affects these representations helps in disambiguation tasks and in understanding the limitations of shared vector spaces.

### Gap in Existing Work
Most literature focuses on overall alignment (isomorphism) between languages or Bilingual Lexicon Induction (BLI) performance. While some work (like Zhang et al. 2019) touches on non-isomorphism, there is less focus on the systematic "pull" that non-shared meanings exert on a word's embedding in a multilingual context. We want to quantify how much a secondary, non-shared meaning "dilutes" the similarity of a word to its translation in another language.

### Our Novel Contribution
We will systematically compare the embedding similarity of monosemous translation pairs versus polysemous pairs where only a subset of meanings are shared. This will provide a "polysemy penalty" metric for cross-lingual alignment.

### Experiment Justification
- **Experiment 1: Baseline Global Alignment**: Establish the expected similarity for "standard" translations to provide a control group.
- **Experiment 2: Polysemy Impact Analysis**: Compare similarity scores for monosemous vs. polysemous words using WordNet (OMW) to define sense counts.
- **Experiment 3: Case Study on "False Friends" and "Partial Friends"**: Deep dive into specific word pairs (like English 'bank' vs French 'banque') to see if embeddings cluster more with the shared or non-shared meanings.

---

## Phase 1: Planning

### Research Question
Do words with similar meanings in different languages have similar embeddings in multilingual models? Specifically, how does polysemy (multiple meanings) affect this similarity when only some meanings are shared?

### Hypothesis Decomposition
1. **H1 (Similarity)**: Translation pairs for monosemous words will exhibit significantly higher cosine similarity than random word pairs in MLLMs.
2. **H2 (Polysemy Interference)**: Translation pairs where one or both words are polysemous will have lower similarity than monosemous pairs, proportional to the number of non-shared senses.
3. **H3 (Dominant Sense)**: The embedding of a polysemous word will be closer to its translation if the shared sense is the "dominant" (most frequent) sense in both languages.

### Proposed Methodology

#### Approach
We will use a modern multilingual model (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` or `xlm-roberta-base`) to extract word embeddings. We will use WordNet (via the OMW sample provided) to categorize words by their polysemy (number of synsets they belong to).

#### Experimental Steps
1. **Setup**: Initialize environment and load models.
2. **Data Processing**: 
    - Extract English-French and English-Spanish pairs from MUSE.
    - Annotate these pairs with "Sense Count" from WordNet/OMW.
    - Identify "Shared Senses" vs "Unique Senses".
3. **Similarity Calculation**:
    - Calculate cosine similarity for all pairs.
    - Group results by polysemy levels (Monosemous-Monosemous, Monosemous-Polysemous, Polysemous-Polysemous).
4. **Statistical Analysis**: 
    - Perform T-tests/ANOVA to see if polysemy significantly reduces similarity.
    - Correlation analysis between (Sense Count) and (Similarity).

### Baselines
- **Random Pairs**: Similarity between non-translation pairs.
- **Monosemous Translations**: The "Gold Standard" for similarity.

### Evaluation Metrics
- **Cosine Similarity**: Primary metric for embedding closeness.
- **CSLS (Cross-Domain Similarity Local Scaling)**: To account for the "hubness" problem in high-dimensional spaces.

### Statistical Analysis Plan
- Correlation (Spearman) between total senses and cosine similarity.
- T-test comparing similarity of Monosemous vs Polysemous pairs.

## Success Criteria
- Successfully quantified the relationship between polysemy and cross-lingual embedding similarity.
- Identified whether "partial" meaning similarity is sufficient for "high" embedding similarity.
