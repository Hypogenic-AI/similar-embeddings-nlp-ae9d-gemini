# Literature Review: Similarity of Word Embeddings Across Languages

## Research Area Overview
The study of cross-lingual word embeddings (CLWE) and multilingual large language models (MLLMs) aims to create a shared vector space where words with similar meanings from different languages are represented by similar vectors. This field has evolved from alignment-based methods for static embeddings (e.g., MUSE) to implicitly aligned representations in pretrained transformers (e.g., mBERT, XLM-R).

## Key Papers

### 1. Emerging Cross-lingual Structure in Pretrained Language Models (2019)
- **Authors**: Wu et al.
- **Key Contribution**: Shows that cross-lingual transfer is possible in MLLMs even without shared vocabulary, suggesting emergent universal latent symmetries.
- **Methodology**: Analyzes MLLMs and shows that monolingual BERT models can be aligned post-hoc effectively.
- **Relevance**: Supports the hypothesis that words with similar meanings have similar embeddings.

### 2. A Survey of Cross-lingual Word Embedding Models (2017)
- **Authors**: Ruder et al.
- **Key Contribution**: Provides a comprehensive typology of CLWE models, comparing their data requirements and objective functions.
- **Relevance**: Foundational for understanding different alignment strategies.

### 3. Concept Space Alignment in Multilingual LLMs (2024)
- **Authors**: Peng & Søgaard
- **Key Contribution**: Evaluates concept alignment in modern LLMs (Llama-2, BLOOMZ, etc.) and finds high-quality linear alignments.
- **Methodology**: Uses WordNet synsets as concepts and Procrustes analysis for alignment.
- **Results**: Generalization works best for typologically similar languages and for abstract concepts.

### 4. Brains and language models converge on a shared conceptual space across different languages (2025)
- **Authors**: Zada et al.
- **Key Contribution**: Finds that both brains and LMs converge on a shared conceptual space regardless of the language.
- **Relevance**: Provides strong cross-disciplinary evidence for the hypothesis.

### 5. Are Girls Neko or Shōjo? (2019)
- **Authors**: Zhang et al.
- **Key Contribution**: Addresses the issue of non-isomorphism in embedding spaces using Iterative Normalization.
- **Relevance**: Relevant to the part of the hypothesis about multiple meanings and how they might affect alignment.

## Common Methodologies
- **Procrustes Analysis**: Finding a linear transformation (rotation) to align two embedding spaces using a seed dictionary.
- **Unsupervised Alignment (MUSE)**: Using adversarial training and Procrustes to align spaces without parallel data.
- **WordNet Alignment**: Using multilingual WordNets (OMW) to define parallel concepts.

## Evaluation Metrics
- **Bilingual Lexicon Induction (BLI)**: Accuracy of translating words using nearest neighbor search in the shared space.
- **Word Similarity**: Correlation with human judgments (e.g., SimLex-999).
- **Cross-lingual Word Similarity**: Measuring similarity between word pairs in different languages.

## Recommended for Our Experiment
- **Datasets**: MUSE dictionaries (en-fr, en-es, en-de), OMW (WordNet) for concept-based evaluation, and False Friends dataset for testing multiple meanings.
- **Baselines**: MUSE unsupervised alignment, Procrustes on mBERT/Llama-2 embeddings.
- **Hypothesis Testing**: Focus on words with multiple meanings (polysemy) using WordNet to identify shared vs. distinct senses.
