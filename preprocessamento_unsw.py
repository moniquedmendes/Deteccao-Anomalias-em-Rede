# preprocessamento_unsw.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def _encode_treino_teste(X_treino, X_teste, colunas_categoricas):
    """
    Codifica colunas categóricas com LabelEncoder treinado SOMENTE no treino.
    Categorias que aparecerem no teste mas não existiam no treino são mapeadas
    para um código extra "desconhecido" (evita vazamento e evita erro do
    LabelEncoder com categoria nunca vista).
    """
    X_treino = X_treino.copy()
    X_teste = X_teste.copy()

    for coluna in colunas_categoricas:
        encoder = LabelEncoder()
        encoder.fit(X_treino[coluna].astype(str))

        mapa = {classe: codigo for codigo, classe in enumerate(encoder.classes_)}
        codigo_desconhecido = len(encoder.classes_)  # categoria nova = código extra

        X_treino[coluna] = X_treino[coluna].astype(str).map(mapa)
        X_teste[coluna] = X_teste[coluna].astype(str).map(mapa).fillna(codigo_desconhecido).astype(int)

    return X_treino, X_teste


def carregar_e_preparar_unsw():
    """
    Versão corrigida: carrega treino e teste do UNSW-NB15 separadamente,
    preservando a divisão original disponibilizada no Kaggle
    (training-set.csv / testing-set.csv), sem redivisão via train_test_split.
    """

    # ======================================
    # 1. CARREGAR — divisão original preservada
    # ======================================
    treino = pd.read_csv("data/UNSW_NB15_training-set.csv")
    teste  = pd.read_csv("data/UNSW_NB15_testing-set.csv")

    print(f"Treino original : {treino.shape}")
    print(f"Teste original  : {teste.shape}")

    proporcao_treino = sum(treino["label"] == 1) / len(treino) * 100
    proporcao_teste  = sum(teste["label"] == 1)  / len(teste)  * 100
    print(f"Proporção de ataques no treino: {proporcao_treino:.2f}%")
    print(f"Proporção de ataques no teste : {proporcao_teste:.2f}%")

    # ======================================
    # 2. EXTRAIR RÓTULO — separadamente
    # ======================================
    y_treino = treino["label"].copy()
    y_teste  = teste["label"].copy()

    print(f"\nDistribuição treino — Normal: {sum(y_treino==0)} | Ataque: {sum(y_treino==1)}")
    print(f"Distribuição teste  — Normal: {sum(y_teste==0)}  | Ataque: {sum(y_teste==1)}")

    # ======================================
    # 3. SEPARAR FEATURES
    # ======================================
    X_treino = treino.drop(["id", "attack_cat", "label"], axis=1)
    X_teste  = teste.drop(["id", "attack_cat", "label"],  axis=1)

    # ======================================
    # 4. ENCODING — fit só no treino
    # ======================================
    colunas_categoricas = [c for c in X_treino.columns
                           if not pd.api.types.is_numeric_dtype(X_treino[c])]
    print("\nColunas categóricas:", colunas_categoricas)

    X_treino, X_teste = _encode_treino_teste(X_treino, X_teste, colunas_categoricas)
    print("Encoding concluído.")

    # ======================================
    # 5. SALVAR NOMES
    # ======================================
    feature_names = list(X_treino.columns)

    # ======================================
    # 6. NORMALIZAÇÃO — fit só no treino
    # ======================================
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste  = scaler.transform(X_teste)
    print("Normalização concluída.")

    print(f"\nTreino: {X_treino.shape} | Teste: {X_teste.shape}")

    return X_treino, X_teste, y_treino, y_teste, feature_names