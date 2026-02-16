import json
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd

def calculate_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def main():
    input_path = 'results/processed_pairs.json'
    output_dir = 'results'
    
    with open(input_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    print("Loading model...")
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    
    print(f"Using device: {device}")
    
    results = []
    
    # To avoid redundant encoding
    all_en_words = list(set(p['en_word'] for p in pairs))
    all_fr_words = list(set(p['fr_word'] for p in pairs))
    
    print(f"Encoding {len(all_en_words)} English words and {len(all_fr_words)} French words...")
    en_embeddings = model.encode(all_en_words, convert_to_numpy=True)
    fr_embeddings = model.encode(all_fr_words, convert_to_numpy=True)
    
    en_emb_dict = dict(zip(all_en_words, en_embeddings))
    fr_emb_dict = dict(zip(all_fr_words, fr_embeddings))
    
    print("Calculating similarities...")
    for p in pairs:
        en_vec = en_emb_dict[p['en_word']]
        fr_vec = fr_emb_dict[p['fr_word']]
        sim = calculate_cosine_similarity(en_vec, fr_vec)
        
        # Categorize
        if p['en_senses'] == 1 and p['fr_senses'] == 1:
            category = 'Monosemous-Shared'
        elif p['shared_senses'] == p['en_senses'] and p['shared_senses'] == p['fr_senses']:
            category = 'Polysemous-Full'
        else:
            category = 'Polysemous-Partial'
            
        results.append({
            **p,
            'similarity': float(sim),
            'category': category
        })
    
    # Random baseline
    print("Generating random baseline...")
    random_sims = []
    for _ in range(len(results)):
        en_word = np.random.choice(all_en_words)
        fr_word = np.random.choice(all_fr_words)
        en_vec = en_emb_dict[en_word]
        fr_vec = fr_emb_dict[fr_word]
        random_sims.append(float(calculate_cosine_similarity(en_vec, fr_vec)))
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'similarity_results.csv'), index=False)
    
    # Summary statistics
    summary = df.groupby('category')['similarity'].agg(['mean', 'std', 'count']).to_dict('index')
    summary['Random-Baseline'] = {
        'mean': float(np.mean(random_sims)),
        'std': float(np.std(random_sims)),
        'count': len(random_sims)
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("Results saved to results/similarity_results.csv and results/summary.json")

if __name__ == "__main__":
    main()
