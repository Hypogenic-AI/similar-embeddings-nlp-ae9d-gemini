# Multilingual Word Embedding Similarity and Polysemy

This project investigates how the similarity of word embeddings in multilingual models is affected by polysemy (multiple meanings), specifically when only some meanings are shared across languages.

## Key Findings
- **Translation pairs remain similar**: Even with non-shared meanings, word pairs are significantly more similar (~0.8 cosine similarity) than random pairs (~0.37).
- **Semantic Interference**: There is a statistically significant negative correlation (ρ = -0.125, p = 0.010) between the number of non-shared senses and embedding similarity.
- **Language Differences**: The interference effect was more pronounced in English-Spanish (ρ = -0.205) than in English-French (ρ = -0.096).

## Methodology
1. **Data**: Used MUSE dictionaries (EN-FR, EN-ES) and Open Multilingual WordNet (OMW) for sense annotation.
2. **Model**: Extracted embeddings from `paraphrase-multilingual-MiniLM-L12-v2`.
3. **Analysis**: Categorized pairs by polysemy overlap and calculated Spearman correlation between non-shared senses and cosine similarity.

## Reproducing Results
1. **Setup**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
2. **Run Experiments**:
   ```bash
   python src/data_prep.py
   python src/embed_and_sim.py
   python src/combined_exp.py
   python src/analysis.py
   ```
3. **View Results**:
   - `REPORT.md`: Comprehensive analysis.
   - `results/plots/`: Visualizations of similarity distributions.
   - `results/combined_results.csv`: Raw data.

## Project Structure
- `src/`: Python scripts for data processing, embedding extraction, and analysis.
- `results/`: CSV files, JSON summaries, and plots.
- `datasets/`: MUSE dictionaries and OMW sample.
- `papers/`: Relevant research papers.
