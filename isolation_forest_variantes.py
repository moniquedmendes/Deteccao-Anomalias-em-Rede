# isolation_forest_variantes.py

import numpy as np
from sklearn.utils import resample
from isolation_forest import treinar_isolation_forest

# ======================================
# EXPERIMENTOS 009 / 010
# TREINO LIMPO — 0% anomalias
# ======================================

def treinar_if_treino_limpo(X_treino, X_teste, y_treino, y_teste,
                             contamination=0.01, random_state=42,
                             nome_dataset="NSL-KDD"):
    """
    Treina o Isolation Forest usando APENAS registros normais no treino.
    Essa é a forma correta de usar o IF: ele aprende só o padrão normal
    e sinaliza qualquer desvio como anomalia.
    """

    print("=" * 50)
    print(f"IF TREINO LIMPO (0% anomalias) — {nome_dataset}")
    print("=" * 50)

    # Filtra apenas os registros normais do treino
    X_treino_limpo = X_treino[y_treino == 0]

    print(f"\nRegistros no treino original : {X_treino.shape[0]}")
    print(f"Registros normais no treino  : {X_treino_limpo.shape[0]}")
    print(f"Registros de ataque removidos: {sum(y_treino == 1)}")
    print(f"Proporção de anomalias no treino: 0%")

    # Treina o IF só com tráfego normal
    modelo, y_pred, escores, metricas = treinar_isolation_forest(
        X_treino_limpo, X_teste, y_teste,
        contamination=contamination,
        random_state=random_state,
        nome_dataset=nome_dataset,
        variante="Treino Limpo (0%)"
    )

    return modelo, y_pred, escores, metricas


# ======================================
# EXPERIMENTOS 011 / 012
# TREINO COM ~3% DE ANOMALIAS
# ======================================

def treinar_if_3pct_anomalias(X_treino, X_teste, y_treino, y_teste,
                               proporcao=0.03, contamination=0.03,
                               random_state=42, nome_dataset="NSL-KDD"):
    """
    Treina o Isolation Forest com uma proporção pequena de ataques
    no conjunto de treino (~3%), simulando um cenário realista de produção
    onde ataques existem mas são raros.
    """

    print("=" * 50)
    print(f"IF TREINO ~{int(proporcao*100)}% ANOMALIAS — {nome_dataset}")
    print("=" * 50)

    # Separa normais e ataques do treino
    X_normal = X_treino[y_treino == 0]
    X_ataque = X_treino[y_treino == 1]

    # Calcula quantos ataques representam ~3% do total de normais
    n_ataques = int(len(X_normal) * proporcao)

    # Amostra aleatória dos ataques
    X_ataques_amostra = resample(
        X_ataque,
        n_samples=n_ataques,
        random_state=random_state,
        replace=False
    )

    # Monta o treino: todos os normais + amostra pequena de ataques
    X_treino_3pct = np.vstack([X_normal, X_ataques_amostra])

    proporcao_real = n_ataques / len(X_treino_3pct) * 100

    print(f"\nRegistros normais no treino  : {len(X_normal)}")
    print(f"Ataques amostrados           : {n_ataques}")
    print(f"Total no treino              : {len(X_treino_3pct)}")
    print(f"Proporção real de anomalias  : {proporcao_real:.2f}%")

    # Treina o IF com contamination ajustado para a proporção real
    modelo, y_pred, escores, metricas = treinar_isolation_forest(
        X_treino_3pct, X_teste, y_teste,
        contamination=contamination,
        random_state=random_state,
        nome_dataset=nome_dataset,
        variante=f"Treino {proporcao_real:.1f}% anomalias"
    )

    return modelo, y_pred, escores, metricas