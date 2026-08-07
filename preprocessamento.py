# preprocessamento.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# NSL-KDD:
# 49 features (contra 41 do NSL-KDD)
# ~2,5 milhões de registros
# 9 categorias de ataque + normal
# Gerado com ferramentas Argus e Bro-IDS
# Tem colunas com tipos mistos, algumas que parecem numéricas mas têm texto


def _encode_treino_teste(X_treino, X_teste, colunas_categoricas):
    """
    Codifica colunas categóricas com LabelEncoder treinado SOMENTE no treino.
    Categorias que aparecerem no teste mas não existiam no treino são mapeadas
    para um código extra "desconhecido" (evita vazamento e evita erro do
    LabelEncoder com categoria nunca vista), depois escrever sobre isso na parte escrita.
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


def carregar_e_preparar(test_size=0.2, random_state=42):

    colunas = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
        "root_shell","su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
        "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "classe",       # rótulo real (normal, neptune, smurf...)
        "dificuldade"   # coluna extra do NSL-KDD, não é feature
    ]

    # ======================================
    # 1. CARREGAR DADOS
    # ======================================
    treino = pd.read_csv("data/KDDTrain+.txt", names=colunas)
    teste  = pd.read_csv("data/KDDTest+.txt",  names=colunas)
    dados  = pd.concat([treino, teste], ignore_index=True)

    print(f"Dataset carregado: {dados.shape}")

    # ======================================
    # 2. DIAGNÓSTICO DA COLUNA CLASSE
    # ======================================
    print("\nValores únicos na coluna classe:")
    print(sorted(dados["classe"].unique()))

    # ======================================
    # 3. EXTRAIR RÓTULO - ANTES DE QUALQUER ENCODING
    # ======================================
    y = dados["classe"].apply(
        lambda x: 0 if str(x).strip().lower() == "normal" else 1
    )

    print("\nDistribuição das classes:")
    print(y.value_counts())
    print(f"Normal: {sum(y==0)} | Ataque: {sum(y==1)}")

    # ======================================
    # 4. SEPARAR FEATURES
    # ======================================
    X = dados.drop(["classe", "dificuldade"], axis=1)

    colunas_categoricas = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if colunas_categoricas:
        print("\nColunas categóricas detectadas:", colunas_categoricas)
    else:
        print("\nNenhuma coluna categórica detectada.")

    # ======================================
    # 5. DIVISÃO TREINO / TESTE (ANTES DO ENCODING E DA NORMALIZAÇÃO)
    # ======================================
    # Fazemos o split aqui, com os dados ainda "crus", para que o encoding e a
    # normalização sejam ajustados (fit) apenas no treino. Isso evita que
    # estatísticas do teste vazem para o pré-processamento.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y      # mantém proporção de classes nos dois conjuntos
    )

    # ======================================
    # 6. ENCODING DAS COLUNAS CATEGÓRICAS (fit só no treino)
    # ======================================
    X_treino, X_teste = _encode_treino_teste(X_treino, X_teste, colunas_categoricas)

    colunas_texto_restantes = [c for c in X_treino.columns if not pd.api.types.is_numeric_dtype(X_treino[c])]
    if colunas_texto_restantes:
        print("  Colunas ainda em texto:", colunas_texto_restantes)
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

   # ======================================
   # esqueci o que ia escrever foi mal
   # ======================================

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
    # label (rótulo)  nenhuma dessas é feature de rede
    X = dados.drop(["id", "attack_cat", "label"], axis=1)

    print(f"\nFeatures utilizadas: {X.shape[1]}")
    print("Colunas:", list(X.columns))

    # UNSW-NB15 tem 3 colunas categóricas: proto, service, state
<<<<<<< HEAD
    colunas_categoricas = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    print("Colunas categóricas detectadas:", colunas_categoricas)
=======
    #TOMAR CUIDADO COM A VERSÃO DO PANDA, POSSIVEL ERRO POR CAUSA DA INVERSÃO, ISSO NOS DOIS PROCESSAMENTOS QUE TEM!!
    for coluna in X.columns:
        if X[coluna].dtype == "object":
            encoder = LabelEncoder()
            X[coluna] = encoder.fit_transform(X[coluna].astype(str))

    colunas_texto = X.dtypes[X.dtypes == "object"]
    if len(colunas_texto) > 0:
        print("Colunas ainda em texto:", list(colunas_texto.index))
    else:
        print("\n Todas as colunas convertidas para numérico.")
>>>>>>> c420f3132513739167c51c92b15c1e38d7489e9d

    # ======================================
    # 5. DIVISÃO TREINO / TESTE (ANTES DO ENCODING E DA NORMALIZAÇÃO)
    # ======================================
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
