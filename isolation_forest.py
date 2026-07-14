# isolation_forest.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import os

# ======================================
# FUNÇÃO PRINCIPAL
# ======================================

def treinar_isolation_forest(X_treino, X_teste, y_teste,
                              contamination=0.1, random_state=42):
    """
    Treina um Isolation Forest e retorna o modelo, predições e métricas.

    Parâmetros:
        X_treino      : array de features para treino (sem rótulos)
        X_teste       : array de features para teste
        y_teste       : rótulos reais do teste (só para avaliação)
        contamination : proporção esperada de anomalias nos dados (0.0 a 0.5)
        random_state  : semente para reprodutibilidade

    Retorna:
        modelo   : modelo treinado
        y_pred   : predições binárias (0=normal, 1=ataque)
        escores  : escores de anomalia brutos para cada registro
        metricas : dicionário com os resultados
    """

    print("=" * 50)
    print("ISOLATION FOREST — NSL-KDD")
    print("=" * 50)

    # ======================================
    # 1. INSTANCIAR E TREINAR O MODELO
    # ======================================
    # Importante: o IF treina SEM y_treino — ele aprende
    # apenas o padrão dos dados, sem saber o que é ataque

    modelo = IsolationForest(
        n_estimators=100,       # número de árvores de isolamento
        contamination=contamination,  # proporção esperada de anomalias
        random_state=random_state,
        n_jobs=-1
    )

    print(f"\nTreinando com contamination={contamination}...")
    modelo.fit(X_treino)       # ← sem y_treino aqui
    print("Treinamento concluído.")

    # ======================================
    # 2. PREDIÇÃO E CONVERSÃO DOS RÓTULOS
    # ======================================
    # O IF retorna:  1 = normal  |  -1 = anomalia
    # Nosso padrão:  0 = normal  |   1 = ataque
    # Precisamos converter para manter consistência com o RF

    predicoes_raw = modelo.predict(X_teste)
    y_pred = np.where(predicoes_raw == 1, 0, 1)

    # Escore de anomalia bruto — quanto menor, mais anômalo
    # Valores negativos indicam anomalia
    escores = modelo.score_samples(X_teste)

    # ======================================
    # 3. MÉTRICAS
    # ======================================
    metricas = {
        "acuracia" : accuracy_score(y_teste, y_pred),
        "precisao" : precision_score(y_teste, y_pred, zero_division=0),
        "recall"   : recall_score(y_teste, y_pred, zero_division=0),
        "f1"       : f1_score(y_teste, y_pred, zero_division=0)
    }

    print("\n--- Resultados ---")
    print(f"Acurácia  : {metricas['acuracia']:.4f}")
    print(f"Precisão  : {metricas['precisao']:.4f}")
    print(f"Recall    : {metricas['recall']:.4f}")
    print(f"F1-score  : {metricas['f1']:.4f}")
    print("\nRelatório completo:")
    print(classification_report(y_teste, y_pred,
                                 target_names=["Normal", "Ataque"],
                                 labels=[0, 1],
                                 zero_division=0))

    # ======================================
    # 4. GRÁFICOS
    # ======================================
    plotar_matriz_confusao(y_teste, y_pred)
    plotar_escores_anomalia(escores, y_teste)

    return modelo, y_pred, escores, metricas


# ======================================
# GRÁFICO — MATRIZ DE CONFUSÃO
# ======================================

def plotar_matriz_confusao(y_teste, y_pred):

    cm = confusion_matrix(y_teste, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Ataque"],
                yticklabels=["Normal", "Ataque"])
    plt.title("Matriz de Confusão — Isolation Forest")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")
    plt.tight_layout()

    os.makedirs("resultados", exist_ok=True)
    plt.savefig("resultados/matriz_confusao_isolation_forest.png", dpi=150)
    plt.show()
    print("Figura salva em: resultados/")


# ======================================
# GRÁFICO — DISTRIBUIÇÃO DOS ESCORES
# ======================================

def plotar_escores_anomalia(escores, y_teste):
    """
    Plota a distribuição dos escores de anomalia separados por classe real.
    Útil para visualizar o quanto o IF separa normal de ataque.
    """

    plt.figure(figsize=(10, 5))

    plt.hist(escores[y_teste == 0], bins=60, alpha=0.6,
             color="steelblue", label="Normal")
    plt.hist(escores[y_teste == 1], bins=60, alpha=0.6,
             color="tomato", label="Ataque")

    plt.title("Distribuição dos Escores de Anomalia — Isolation Forest")
    plt.xlabel("Escore de Anomalia (quanto menor, mais suspeito)")
    plt.ylabel("Quantidade de Registros")
    plt.legend()
    plt.tight_layout()

    os.makedirs("resultados", exist_ok=True)
    plt.savefig("resultados/escores_anomalia_if.png", dpi=150)
    plt.show()
    print("Figura salva em: resultados/")