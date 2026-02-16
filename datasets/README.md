# Downloaded Datasets

## MUSE Ground-truth Dictionaries
- **Source**: Facebook Research (MUSE)
- **Format**: TXT (word pairs)
- **Language Pairs**: en-fr, en-es, en-de
- **Location**: datasets/en-fr.txt, etc.
- **Description**: Standard bilingual dictionaries for word translation.

## SimLex-999
- **Source**: tasksource/simlex (HuggingFace)
- **Format**: Dataset
- **Language**: English (baseline)
- **Description**: Word similarity dataset.

## Potential Datasets (to be downloaded/referenced)
- **Multi-SimLex**: Multilingual lexical similarity.
- **MCL-WiC**: Multilingual Contextual Word Similarity (SemEval-2021 Task 2).
- **XL-WSD**: Cross-lingual Word Sense Disambiguation.

## Download Instructions
MUSE dictionaries were downloaded using wget.
SimLex-999 can be loaded using `datasets.load_dataset('tasksource/simlex')`.
