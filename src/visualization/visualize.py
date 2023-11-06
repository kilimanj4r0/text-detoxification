import matplotlib.pyplot as plt
import pandas as pd


# The data should be downloaded and preprocessed
raw_data_path = '../../data/raw/filtered.tsv'
preprocessed_data_path = '../../data/raw/preprocessed_filtered.csv'
df = pd.read_csv(raw_data_path, sep='\t', index_col=0)
preprocessed_df = pd.read_csv(preprocessed_data_path, index_col=0)

sorted_toxicities = sorted(list(zip(list(df['ref_tox']), list(df['trn_tox']))), key=lambda x: x[0])
indexes = list(range(len(sorted_toxicities)))

plt.plot(indexes, sorted_toxicities, label=('Toxicity Reference Values', 'Toxicity Translation Values'))
plt.xlabel('Index')
plt.ylabel('Toxicity')
plt.title('ref_tox and trn_tox Sorted Increasingly')
plt.legend()
plt.savefig('../../reports/figures/toxicities_before.png')


sorted_toxicities = sorted(list(zip(list(preprocessed_df['ref_tox']), list(preprocessed_df['trn_tox']))), key=lambda x: x[0])
indexes = list(range(len(sorted_toxicities)))

plt.plot(indexes, sorted_toxicities, label=('Toxicity Reference Values', 'Toxicity Translation Values'))
plt.xlabel('Index')
plt.ylabel('Toxicity')
plt.title('ref_tox and trn_tox Sorted Increasingly')
plt.legend()
plt.savefig('../../reports/figures/toxicities_after.png')
