# random_forest.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import os

# ======================================
# FUNÇÃO PRINCIPAL
# ======================================

def treinar_random_forest(X_treino, X_teste, y_treino, y_teste,
                           feature_names=None,        # novo parâmetro
                           n_estimators=100, random_state=42):

    print("=" * 50)
    print("RANDOM FOREST — NSL-KDD")
    print("=" * 50)

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    print(f"\nTreinando com {n_estimators} árvores...")
    modelo.fit(X_treino, y_treino)
    print("Treinamento concluído.")

    y_pred = modelo.predict(X_teste)

    metricas = {
        "acuracia" : accuracy_score(y_teste, y_pred),
        "precisao" : precision_score(y_teste, y_pred),
        "recall"   : recall_score(y_teste, y_pred),
        "f1"       : f1_score(y_teste, y_pred)
    }

    print("\n--- Resultados ---")
    print(f"Acurácia  : {metricas['acuracia']:.4f}")
    print(f"Precisão  : {metricas['precisao']:.4f}")
    print(f"Recall    : {metricas['recall']:.4f}")
    print(f"F1-score  : {metricas['f1']:.4f}")
    print("\nRelatório completo:")
    print(classification_report(y_teste, y_pred,
                                target_names=["Normal", "Ataque"],
                                labels=[0, 1]))

    plotar_matriz_confusao(y_teste, y_pred, nome_modelo="Random Forest")
    plotar_importancia_features(modelo, feature_names=feature_names)  # passa os nomes

    return modelo, y_pred, metricas


# ======================================
# GRÁFICO — MATRIZ DE CONFUSÃO
# ======================================

def plotar_matriz_confusao(y_teste, y_pred, nome_modelo="Modelo"):

    cm = confusion_matrix(y_teste, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Ataque"],
                yticklabels=["Normal", "Ataque"])
    plt.title(f"Matriz de Confusão — {nome_modelo}")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")
    plt.tight_layout()

    os.makedirs("resultados", exist_ok=True)
    plt.savefig(f"resultados/matriz_confusao_{nome_modelo.lower().replace(' ', '_')}.png",
                dpi=150)
    plt.show()
    print("Figura salva em: resultados/")


# ======================================
# GRÁFICO — IMPORTÂNCIA DAS FEATURES
# ======================================

def plotar_importancia_features(modelo, feature_names=None, n_top=15):

    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1][:n_top]

    # Usa os nomes reais se disponíveis, senão usa índices
    if feature_names is not None:
        labels = [feature_names[i] for i in indices]
    else:
        labels = [f"f{i}" for i in indices]

    plt.figure(figsize=(12, 5))
    plt.bar(range(n_top), importancias[indices], color="steelblue")
    plt.xticks(range(n_top), labels, rotation=45, ha="right")
    plt.title(f"Top {n_top} Features Mais Importantes — Random Forest")
    plt.xlabel("Feature")
    plt.ylabel("Importância")
    plt.tight_layout()

    os.makedirs("resultados", exist_ok=True)
    plt.savefig("resultados/importancia_features_rf.png", dpi=150)
    plt.show()
    print("Figura salva em: resultados/")