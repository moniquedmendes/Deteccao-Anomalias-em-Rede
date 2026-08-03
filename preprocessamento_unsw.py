# preprocessamento_unsw.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def carregar_e_preparar_unsw(test_size=0.2, random_state=42):
    """
    Carrega e prepara o dataset UNSW-NB15 para treinamento.

    Retorna:
        X_treino, X_teste, y_treino, y_teste, feature_names
    """

    # ======================================
    # 1. CARREGAR DADOS
    # ======================================
    treino = pd.read_csv("data/UNSW_NB15_training-set.csv")
    teste  = pd.read_csv("data/UNSW_NB15_testing-set.csv")
    dados  = pd.concat([treino, teste], ignore_index=True)

    print(f"Dataset carregado: {dados.shape}")

    # ======================================
    # 2. DIAGNÓSTICO DO RÓTULO
    # ======================================
    print("\nValores únicos em label:", dados["label"].unique())
    print("Valores únicos em attack_cat:", dados["attack_cat"].unique())

    # ======================================
    # 3. EXTRAIR RÓTULO
    # ======================================
    # Diferente do NSL-KDD, o rótulo já é binário:
    # 0 = normal | 1 = ataque
    y = dados["label"].copy()

    print("\nDistribuição das classes:")
    print(y.value_counts())
    print(f"Normal: {sum(y==0)} | Ataque: {sum(y==1)}")

    # ======================================
    # 4. SEPARAR FEATURES
    # ======================================
    # Remover: id (identificador), attack_cat (categoria textual),
    # label (rótulo) — nenhuma dessas é feature de rede
    X = dados.drop(["id", "attack_cat", "label"], axis=1)

    print(f"\nFeatures utilizadas: {X.shape[1]}")
    print("Colunas:", list(X.columns))

    # ======================================
    # 5. ENCODING DAS COLUNAS CATEGÓRICAS
    # ======================================
    # UNSW-NB15 tem 3 colunas categóricas: proto, service, state
    for coluna in X.columns:
        if X[coluna].dtype == "object":
            encoder = LabelEncoder()
            X[coluna] = encoder.fit_transform(X[coluna].astype(str))

    colunas_texto = X.dtypes[X.dtypes == "object"]
    if len(colunas_texto) > 0:
        print(" Colunas ainda em texto:", list(colunas_texto.index))
    else:
        print("\n Todas as colunas convertidas para numérico.")

    # ======================================
    # 6. SALVAR NOMES DAS FEATURES
    # ======================================
    feature_names = list(X.columns)

    # ======================================
    # 7. NORMALIZAÇÃO
    # ======================================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(" Normalização concluída.")

    # ======================================
    # 8. DIVISÃO TREINO / TESTE
    # ======================================
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\nTreino: {X_treino.shape} | Teste: {X_teste.shape}")
    print(f"Distribuição treino — Normal: {sum(y_treino==0)} | Ataque: {sum(y_treino==1)}")
    print(f"Distribuição teste  — Normal: {sum(y_teste==0)}  | Ataque: {sum(y_teste==1)}")

    return X_treino, X_teste, y_treino, y_teste, feature_names