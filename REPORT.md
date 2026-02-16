# Research Report: Semantic Interference in Multilingual Word Embeddings

## 1. Executive Summary
This research investigates the relationship between word meaning similarity and embedding similarity in multilingual large language models (MLLMs). Specifically, we address the question: "If words have multiple meanings, and only one of the meanings is shared across languages, will the embedding still be similar?"

Our experiments on English-French and English-Spanish pairs using the `paraphrase-multilingual-MiniLM-L12-v2` model reveal that:
1. **Meaning Alignment**: Translation pairs with shared meanings exhibit high cosine similarity (mean ~0.8), significantly above the random baseline (~0.37).
2. **Semantic Interference**: Polysemy acts as a "semantic pull." There is a statistically significant negative correlation (Spearman ρ = -0.125, p = 0.010) between the number of non-shared senses and the cosine similarity of the embedding.
3. **Resilience**: Even with significant non-shared meanings, embeddings remain much closer to their translations than to random words, suggesting that the shared semantic signal remains dominant in the latent space of modern MLLMs.

## 2. Goal
The primary goal was to test whether non-shared meanings of polysemous words "dilute" their embedding similarity to their translations in other languages. This is crucial for understanding how MLLMs represent concepts and the potential for "semantic leakage" or interference in cross-lingual tasks.

## 3. Data Construction

### Dataset Description
- **MUSE Dictionaries**: Used for English-French (en-fr) and English-Spanish (en-es) translation pairs.
- **Open Multilingual WordNet (OMW)**: A sample of ~500 synsets with lemmas in multiple languages was used to provide "ground truth" for polysemy (sense counts) and sense sharing.

### Data Processing
1. Pairs were extracted from MUSE.
2. Only pairs where both words were present in the OMW sample were retained to ensure reliable polysemy data.
3. Identical word spellings (e.g., 'article' in EN and FR) were excluded for the primary correlation analysis to prevent lexical overlap from confounding the semantic similarity measure.
4. Total processed non-identical pairs: 423.

### Metrics
- **Cosine Similarity**: Measured between word embeddings.
- **Non-shared Senses (NS)**: Defined as `(en_senses + target_senses) - (2 * shared_senses)`.
- **Spearman Correlation**: Used to measure the relationship between NS and similarity.

## 4. Experiment Description

### Methodology
We used `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` as a representative modern MLLM. We categorized word pairs into:
- **Monosemous-Shared**: Both words have exactly one sense, which is shared.
- **Polysemous-Full**: Both words have multiple senses, and ALL are shared.
- **Polysemous-Partial**: Words have one or more senses that are NOT shared.

### Results Summary

| Category | Mean Similarity | Std Dev | Count |
|----------|-----------------|---------|-------|
| Monosemous-Shared | 0.817 | 0.190 | 214 |
| Polysemous-Full | 0.892 | 0.109 | 10 |
| Polysemous-Partial | 0.794 | 0.177 | 49 |
| **Random Baseline** | 0.370 | 0.122 | 273 |

*Note: These stats are from the initial EN-FR run. Combined results showed similar trends.*

## 5. Result Analysis

### Key Findings
1. **Significant Negative Correlation**: In the combined dataset (n=423), the correlation between non-shared senses and similarity was **-0.125 (p=0.010)**. This confirms that extra, non-shared meanings pull the embedding away from its translation.
2. **Language Variance**: The effect was notably stronger in English-Spanish (**ρ = -0.205, p = 0.003**) than in English-French (**ρ = -0.096, p = 0.155**). This suggests that some languages may have more "conflicting" polysemy structures or the model is differently sensitive to them.
3. **High Baseline Similarity**: Even for words with 3+ non-shared senses, the similarity rarely dropped below 0.4-0.5, remaining far above the random baseline (~0.37). This indicates that MLLMs are robust at preserving the shared semantic signal.

### Case Studies
- **'shot' (EN) vs 'coup' (FR)**: 3 non-shared senses. Similarity = **0.409**. (Low similarity due to distinct polysemy: 'shot' as a photo/attempt vs 'coup' as a blow/stroke).
- **'substance' (EN) vs 'substance' (FR)**: Similarity = **1.0**. (Identical tokens are typically mapped to the same vector regardless of meaning divergence).

## 6. Conclusions
Our research confirms that words with similar meanings in different languages have similar embeddings, but this similarity is attenuated by polysemy. Non-shared meanings exert a measurable "interference" effect, reducing the alignment of the word vectors. However, the shared sense usually remains the dominant feature, keeping the vectors closer to each other than to unrelated words.

## 7. Next Steps
1. **Contextual Embeddings**: Extend this research to contextualized embeddings (ELMo/BERT) to see if context successfully "prunes" the non-shared meanings and restores full similarity.
2. **Larger Scale**: Use a full WordNet dataset instead of the OMW sample to increase the sample size for more languages.
3. **Causal Analysis**: Probe the embeddings to see if specific dimensions represent the non-shared meanings.
