import json
import collections
import os

def load_omw(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    en_poly = collections.defaultdict(list)
    fr_poly = collections.defaultdict(list)
    
    for entry in data:
        synset = entry['synset']
        for word in entry.get('en', []):
            en_poly[word.lower()].append(synset)
        for word in entry.get('fra', []):
            fr_poly[word.lower()].append(synset)
            
    return en_poly, fr_poly

def load_muse(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0].lower(), parts[1].lower()))
    return pairs

def main():
    omw_path = 'datasets/omw_sample.json'
    muse_path = 'datasets/en-fr.txt'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading OMW...")
    en_poly, fr_poly = load_omw(omw_path)
    
    print("Loading MUSE...")
    pairs = load_muse(muse_path)
    
    print(f"Total MUSE pairs: {len(pairs)}")
    
    processed_pairs = []
    for en_word, fr_word in pairs:
        if en_word in en_poly and fr_word in fr_poly:
            # Check if they share at least one synset
            en_synsets = set(en_poly[en_word])
            fr_synsets = set(fr_poly[fr_word])
            shared_synsets = en_synsets.intersection(fr_synsets)
            
            processed_pairs.append({
                'en_word': en_word,
                'fr_word': fr_word,
                'en_senses': len(en_synsets),
                'fr_senses': len(fr_synsets),
                'shared_senses': len(shared_synsets),
                'en_synsets': list(en_synsets),
                'fr_synsets': list(fr_synsets),
                'shared_synsets_list': list(shared_synsets)
            })
            
    print(f"Pairs with polysemy info: {len(processed_pairs)}")
    
    with open(os.path.join(output_dir, 'processed_pairs.json'), 'w', encoding='utf-8') as f:
        json.dump(processed_pairs, f, indent=2, ensure_ascii=False)
    
    # Also save some statistics
    stats = {
        'total_muse_pairs': len(pairs),
        'pairs_with_info': len(processed_pairs),
        'monosemous_pairs': len([p for p in processed_pairs if p['en_senses'] == 1 and p['fr_senses'] == 1]),
        'polysemous_pairs': len([p for p in processed_pairs if p['en_senses'] > 1 or p['fr_senses'] > 1])
    }
    
    with open(os.path.join(output_dir, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
