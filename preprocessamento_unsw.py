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
    # label (rótulo) nenhuma dessas é feature de rede
    X = dados.drop(["id", "attack_cat", "label"], axis=1)

    print(f"\nFeatures utilizadas: {X.shape[1]}")
    print("Colunas:", list(X.columns))

    # UNSW-NB15 tem 3 colunas categóricas: proto, service, state
    colunas_categoricas = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    print("Colunas categóricas detectadas:", colunas_categoricas)

    # ======================================
    # 5. DIVISÃO TREINO / TESTE (ANTES DO ENCODING E DA NORMALIZAÇÃO)
    # ======================================
    # O split é feito aqui, com os dados ainda "crus", para que o LabelEncoder
    # e o StandardScaler sejam ajustados (fit) somente com o treino o
    # que estatísticas do teste vazem para o pré-processamento.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ======================================
    # 6. ENCODING DAS COLUNAS CATEGÓRICAS (fit só no treino)
    # ======================================
    X_treino, X_teste = _encode_treino_teste(X_treino, X_teste, colunas_categoricas)

    colunas_texto_restantes = [c for c in X_treino.columns if not pd.api.types.is_numeric_dtype(X_treino[c])]
    if colunas_texto_restantes:
        print(" Colunas ainda em texto:", colunas_texto_restantes)
    else:
        print("\n Todas as colunas convertidas para numérico.")

    # ======================================
    # 7. SALVAR NOMES DAS FEATURES
    # ======================================
    feature_names = list(X_treino.columns)

    # ======================================
    # 8. NORMALIZAÇÃO (fit só no treino)
    # ======================================
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)

    print(" Normalização concluída.")

    print(f"\nTreino: {X_treino.shape} | Teste: {X_teste.shape}")
    print(f"Distribuição treino - Normal: {sum(y_treino==0)} | Ataque: {sum(y_treino==1)}")
    print(f"Distribuição teste  - Normal: {sum(y_teste==0)}  | Ataque: {sum(y_teste==1)}")

    return X_treino, X_teste, y_treino, y_teste, feature_names
