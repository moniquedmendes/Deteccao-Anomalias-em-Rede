#ARQUIVO FEITO PRA ESTUDAR O DATASET INSW-NB15!!!!


# diagnostico_unsw.py
import pandas as pd

# Tenta carregar com cabeçalho primeiro
df = pd.read_csv("data/UNSW_NB15_training-set.csv")

print("Shape:", df.shape)
print("\nPrimeiras colunas:")
print(df.columns.tolist())
print("\nTipos de dados:")
print(df.dtypes)
print("\nPrimeiras 3 linhas:")
print(df.head(3))
print("\nValores únicos na coluna de rótulo:")
# tenta as colunas mais prováveis
for col in ["label", "Label", "attack_cat", "class", "classe"]:
    if col in df.columns:
        print(f"\n{col}:", df[col].unique())