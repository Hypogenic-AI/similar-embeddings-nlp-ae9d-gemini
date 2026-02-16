import pandas as pd
from scipy import stats

df = pd.read_csv('results/combined_results.csv')
df = df[df['en_word'] != df['target_word']]
df['ns'] = df['en_senses'] + df['target_senses'] - 2*df['shared_senses']

for lang in ['fr', 'es']:
    sub = df[df['lang'] == lang]
    if len(sub) > 0:
        c, p = stats.spearmanr(sub['ns'], sub['similarity'])
        print(f"{lang}: corr={c:.4f}, p={p:.4f}, count={len(sub)}")
