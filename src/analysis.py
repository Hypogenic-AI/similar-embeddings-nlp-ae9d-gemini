import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def main():
    results_path = 'results/combined_results.csv'
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(results_path)
    # Exclude identical words for cleaner meaning analysis
    df_diff = df[df['en_word'] != df['target_word']]
    
    # 1. Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_diff, x='similarity', hue='category', fill=True, common_norm=False)
    plt.title('Distribution of Cosine Similarity by Polysemy Category (Non-Identical Words)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.savefig(os.path.join(output_dir, 'similarity_distribution_combined.png'))
    plt.close()
    
    # 2. Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_diff, x='category', y='similarity')
    plt.title('Cosine Similarity by Polysemy Category (Non-Identical Words)')
    plt.savefig(os.path.join(output_dir, 'similarity_boxplot_combined.png'))
    plt.close()
    
    # 3. Statistical Tests
    mono = df_diff[df_diff['category'] == 'Monosemous-Shared']['similarity']
    partial = df_diff[df_diff['category'] == 'Polysemous-Partial']['similarity']
    
    t_stat, p_val = stats.ttest_ind(mono, partial, equal_var=False)
    
    # 4. Correlation Analysis
    df_diff['total_senses'] = df_diff['en_senses'] + df_diff['target_senses']
    df_diff['non_shared_senses'] = (df_diff['en_senses'] + df_diff['target_senses']) - (2 * df_diff['shared_senses'])
    
    corr, corr_p = stats.spearmanr(df_diff['non_shared_senses'], df_diff['similarity'])
    
    analysis_results = {
        'ttest_mono_vs_partial': {
            't_stat': float(t_stat),
            'p_val': float(p_val)
        },
        'spearman_non_shared_senses_vs_similarity': {
            'correlation': float(corr),
            'p_val': float(corr_p)
        }
    }
    
    # Language specific analysis
    for lang in df_diff['lang'].unique():
        sub = df_diff[df_diff['lang'] == lang]
        c, p = stats.spearmanr(sub['non_shared_senses'], sub['similarity'])
        analysis_results[f'spearman_{lang}'] = {'correlation': float(c), 'p_val': float(p)}
    
    with open('results/analysis_results_combined.json', 'w') as f:
        import json
        json.dump(analysis_results, f, indent=2)
        
    print("Analysis complete. Results saved to results/analysis_results_combined.json and results/plots/")

if __name__ == "__main__":
    main()
