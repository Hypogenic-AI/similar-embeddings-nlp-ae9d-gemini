# Resources Catalog

## Papers
Total papers: 8

| Title | Year | File | Key Info |
|-------|------|------|----------|
| Emerging Cross-lingual Structure | 2019 | papers/Emerging_Cross_lingual_Structure.pdf | Shared parameters and latent symmetries. |
| A Survey of Cross-lingual Word Embeddings | 2017 | papers/Survey_Cross_lingual_Word_Embeddings.pdf | Typology of CLWE models. |
| Unsupervised Multilingual Word Embeddings | 2018 | papers/Unsupervised_Multilingual_Word_Embeddings.pdf | Framework exploiting interdependencies. |
| Are Girls Neko or Shōjo? | 2019 | papers/Are_Girls_Neko_or_Shojo.pdf | Non-isomorphic alignment. |
| Multi-SimLex | 2020 | papers/Multi_SimLex.pdf | Large-scale semantic similarity benchmark. |
| Word Translation Without Parallel Data | 2017 | papers/Word_Translation_Without_Parallel_Data.pdf | MUSE unsupervised method. |
| Concept Space Alignment in Multilingual LLMs | 2024 | papers/Concept_Space_Alignment.pdf | Alignment of WordNet concepts in LLMs. |
| Brains and LMs Convergence | 2025 | papers/Brains_and_LMs_Convergence.pdf | Shared conceptual space across brains and LMs. |

## Datasets

| Name | Source | File/Location | Description |
|------|--------|---------------|-------------|
| MUSE en-fr | Facebook | datasets/en-fr.txt | Bilingual dictionary for English-French. |
| MUSE en-es | Facebook | datasets/en-es.txt | Bilingual dictionary for English-Spanish. |
| MUSE en-de | Facebook | datasets/en-de.txt | Bilingual dictionary for English-German. |
| OMW Sample | NLTK (WordNet) | datasets/omw_sample.json | 500 WordNet synsets with lemmas in multiple languages. |
| False Friends | dhfbk/falsefriends | code/falsefriends/data | Annotated cognates and false friends (IT-EN, IT-FR). |
| SimLex-999 | tasksource/simlex | (HuggingFace) | Word similarity baseline. |

## Code Repositories

| Name | URL | Location | Purpose |
|------|-----|----------|---------|
| Multilingual Latent Concepts | github.com/qcri/multilingual-latent-concepts | code/multilingual-latent-concepts | Concept alignment and activation extraction. |
| False Friends | github.com/dhfbk/falsefriends | code/falsefriends | False friend detection and dataset. |
| NeuroX | (Included in concepts repo) | code/multilingual-latent-concepts/NeuroX | Interpretation of neural representations. |

## Recommendations for Experiment Design
1. **Primary Dataset**: Use MUSE dictionaries for basic alignment and BLI tasks.
2. **Concept Evaluation**: Use the OMW synsets to measure how well specific concepts align across languages.
3. **Polysemy Check**: Use the False Friends dataset and manually selected polysemous words (e.g., 'bank') to test the "multiple meanings" part of the hypothesis.
4. **Model Choice**: Compare mBERT (masked LM) with Llama-2/3 (causal LM) using the extraction methods from the Peng & Søgaard paper.
