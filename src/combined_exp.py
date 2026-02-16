import json
import collections
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd
from scipy import stats

def load_omw(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    en_poly = collections.defaultdict(list)
    fr_poly = collections.defaultdict(list)
    es_poly = collections.defaultdict(list)
    
    for entry in data:
        synset = entry['synset']
        for word in entry.get('en', []):
            en_poly[word.lower()].append(synset)
        for word in entry.get('fra', []):
            fr_poly[word.lower()].append(synset)
        for word in entry.get('spa', []):
            es_poly[word.lower()].append(synset)
            
    return en_poly, fr_poly, es_poly

def load_muse(file_path):
    pairs = []
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0].lower(), parts[1].lower()))
    return pairs

def calculate_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def main():
    omw_path = 'datasets/omw_sample.json'
    en_fr_path = 'datasets/en-fr.txt'
    en_es_path = 'datasets/en-es.txt'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading OMW...")
    en_poly, fr_poly, es_poly = load_omw(omw_path)
    
    print("Loading MUSE...")
    fr_pairs = load_muse(en_fr_path)
    es_pairs = load_muse(en_es_path)
    
    all_processed = []
    
    for lang, pairs, poly_dict in [('fr', fr_pairs, fr_poly), ('es', es_pairs, es_poly)]:
        print(f"Processing English-{lang}...")
        for en_word, target_word in pairs:
            if en_word in en_poly and target_word in poly_dict:
                en_synsets = set(en_poly[en_word])
                target_synsets = set(poly_dict[target_word])
                shared = en_synsets.intersection(target_synsets)
                
                all_processed.append({
                    'en_word': en_word,
                    'target_word': target_word,
                    'lang': lang,
                    'en_senses': len(en_synsets),
                    'target_senses': len(target_synsets),
                    'shared_senses': len(shared)
                })
    
    print(f"Total processed pairs: {len(all_processed)}")
    
    print("Loading model...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Batch encode all unique words
    en_words = list(set(p['en_word'] for p in all_processed))
    target_words = list(set(p['target_word'] for p in all_processed))
    
    print("Encoding words...")
    en_emb = dict(zip(en_words, model.encode(en_words)))
    target_emb = dict(zip(target_words, model.encode(target_words)))
    
    print("Calculating similarities...")
    for p in all_processed:
        sim = calculate_cosine_similarity(en_emb[p['en_word']], target_emb[p['target_word']])
        p['similarity'] = float(sim)
        
        # Category
        if p['en_senses'] == 1 and p['target_senses'] == 1:
            p['category'] = 'Monosemous-Shared'
        elif p['shared_senses'] == p['en_senses'] and p['shared_senses'] == p['target_senses']:
            p['category'] = 'Polysemous-Full'
        else:
            p['category'] = 'Polysemous-Partial'
    
    df = pd.DataFrame(all_processed)
    df.to_csv(os.path.join(output_dir, 'combined_results.csv'), index=False)
    
    # Analysis excluding identical words
    df_diff = df[df['en_word'] != df['target_word']]
    df_diff['non_shared_senses'] = (df_diff['en_senses'] + df_diff['target_senses']) - (2 * df_diff['shared_senses'])
    
    corr, p_val = stats.spearmanr(df_diff['non_shared_senses'], df_diff['similarity'])
    print(f"Non-identical pairs: Correlation={corr}, p={p_val}, count={len(df_diff)}")
    
    # Save summary
    summary = df.groupby('category')['similarity'].agg(['mean', 'std', 'count']).to_dict('index')
    with open(os.path.join(output_dir, 'combined_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
