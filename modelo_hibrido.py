# modelo_hibrido.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import os

# ======================================
# FUNÇÃO PRINCIPAL
# ======================================

def treinar_modelo_hibrido(y_pred_rf, escores_if, y_teste,
                            threshold=-0.45):
    """
    Combina as saídas do Random Forest e do Isolation Forest.

    Parâmetros:
        y_pred_rf  : predições binárias do Random Forest (0 ou 1)
        escores_if : escores de anomalia do Isolation Forest
        y_teste    : rótulos reais para avaliação
        threshold  : limiar do escore IF abaixo do qual um registro
                     é considerado anômalo (padrão: -0.45)

    Retorna:
        y_pred_hibrido : predições finais combinadas
        metricas       : dicionário com os resultados
    """

    print("=" * 50)
    print("MODELO HÍBRIDO — NSL-KDD")
    print(f"Threshold IF: {threshold}")
    print("=" * 50)

    # ======================================
    # 1. REGRA DE COMBINAÇÃO
    # ======================================
    # Um registro é ataque se:
    #   - RF disse ataque (1), OU
    #   - RF disse normal (0) MAS IF deu escore abaixo do threshold

    y_pred_hibrido = np.where(
        (y_pred_rf == 1) | (escores_if < threshold),
        1,  # ataque
        0   # normal
    )

    # ======================================
    # 2. MÉTRICAS
    # ======================================
    metricas = {
        "acuracia" : accuracy_score(y_teste, y_pred_hibrido),
        "precisao" : precision_score(y_teste, y_pred_hibrido, zero_division=0),
        "recall"   : recall_score(y_teste, y_pred_hibrido, zero_division=0),
        "f1"       : f1_score(y_teste, y_pred_hibrido, zero_division=0)
    }

    print("\n--- Resultados ---")
    print(f"Acurácia  : {metricas['acuracia']:.4f}")
    print(f"Precisão  : {metricas['precisao']:.4f}")
    print(f"Recall    : {metricas['recall']:.4f}")
    print(f"F1-score  : {metricas['f1']:.4f}")
    print("\nRelatório completo:")
    print(classification_report(y_teste, y_pred_hibrido,
                                 target_names=["Normal", "Ataque"],
                                 labels=[0, 1],
                                 zero_division=0))

    # ======================================
    # 3. MATRIZ DE CONFUSÃO
    # ======================================
    plotar_matriz_confusao(y_teste, y_pred_hibrido)

    return y_pred_hibrido, metricas


# ======================================
# ANÁLISE DE THRESHOLDS
# ======================================

def analisar_thresholds(y_pred_rf, escores_if, y_teste,
                         thresholds=None):
    """
    Testa diferentes valores de threshold e mostra o impacto
    nas métricas. Útil para escolher o melhor threshold.
    """

    if thresholds is None:
        thresholds = [-0.65, -0.60, -0.55, -0.50, -0.45, -0.40, -0.35]

    print("\n--- Análise de Thresholds ---")
    print(f"{'Threshold':>12} | {'Acurácia':>9} | {'Precisão':>9} | "
          f"{'Recall':>9} | {'F1-score':>9}")
    print("-" * 60)

    resultados = []

    for t in thresholds:
        y_pred = np.where(
            (y_pred_rf == 1) | (escores_if < t),
            1, 0
        )
        resultado = {
            "threshold" : t,
            "acuracia"  : accuracy_score(y_teste, y_pred),
            "precisao"  : precision_score(y_teste, y_pred, zero_division=0),
            "recall"    : recall_score(y_teste, y_pred, zero_division=0),
            "f1"        : f1_score(y_teste, y_pred, zero_division=0)
        }
        resultados.append(resultado)

        print(f"{t:>12.2f} | {resultado['acuracia']:>9.4f} | "
              f"{resultado['precisao']:>9.4f} | {resultado['recall']:>9.4f} | "
              f"{resultado['f1']:>9.4f}")

    # Plota F1 por threshold
    plotar_thresholds(thresholds, resultados)

    return resultados


# ======================================
# GRÁFICO — MATRIZ DE CONFUSÃO
# ======================================

def plotar_matriz_confusao(y_teste, y_pred):

    cm = confusion_matrix(y_teste, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Ataque"],
                yticklabels=["Normal", "Ataque"])
    plt.title("Matriz de Confusão — Modelo Híbrido")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")
    plt.tight_layout()

    os.makedirs("resultados", exist_ok=True)
    plt.savefig("resultados/matriz_confusao_modelo_hibrido.png", dpi=150)
    plt.show()
    print("Figura salva em: resultados/")


# ======================================
# GRÁFICO — F1 POR THRESHOLD
# ======================================

def plotar_thresholds(thresholds, resultados):

    f1s      = [r["f1"]      for r in resultados]
    recalls  = [r["recall"]  for r in resultados]
    precisoes = [r["precisao"] for r in resultados]

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, f1s,       marker="o", label="F1-score",  color="steelblue")
    plt.plot(thresholds, recalls,   marker="s", label="Recall",    color="tomato")
    plt.plot(thresholds, precisoes, marker="^", label="Precisão",  color="seagreen")

    plt.title("Impacto do Threshold nas Métricas — Modelo Híbrido")
    plt.xlabel("Threshold (escore IF)")
    plt.ylabel("Valor da Métrica")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("resultados", exist_ok=True)
    plt.savefig("resultados/thresholds_modelo_hibrido.png", dpi=150)
    plt.show()
    print("Figura salva em: resultados/")