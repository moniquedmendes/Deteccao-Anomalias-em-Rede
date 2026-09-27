# validacao_cruzada_if.py

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import os

def validacao_cruzada_if(X, y, variante="limpo",
                          contamination=0.1,
                          proporcao_anomalias=0.03,
                          n_splits=5, random_state=42,
                          nome_dataset="NSL-KDD"):
    """
    Validação cruzada estratificada para o Isolation Forest.

    Parâmetros:
        X               : array completo de features (sem split prévio)
        y               : array completo de rótulos
        variante        : "limpo" (0% anomalias) ou "3pct" (~3% anomalias)
        contamination   : parâmetro do IF
        proporcao_anomalias : usado só na variante "3pct"
        n_splits        : número de folds (padrão 5)
        nome_dataset    : para o título dos prints
    """

    print("=" * 55)
    print(f"VALIDAÇÃO CRUZADA — IF {variante.upper()} — {nome_dataset}")
    print(f"Folds: {n_splits} | contamination: {contamination}")
    print("=" * 55)

    X = np.array(X)
    y = np.array(y)

    # StratifiedKFold mantém a proporção de classes em cada fold
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=random_state)

    resultados = []

    for fold, (idx_treino, idx_teste) in enumerate(skf.split(X, y), 1):

        X_treino_fold = X[idx_treino]
        y_treino_fold = y[idx_treino]
        X_teste_fold  = X[idx_teste]
        y_teste_fold  = y[idx_teste]

        # Prepara o treino de acordo com a variante
        if variante == "limpo":
            # Só registros normais no treino
            X_treino_if = X_treino_fold[y_treino_fold == 0]

        elif variante == "3pct":
            # Normais + ~3% de ataques amostrados
            from sklearn.utils import resample
            X_normal = X_treino_fold[y_treino_fold == 0]
            X_ataque = X_treino_fold[y_treino_fold == 1]
            n_ataques = int(len(X_normal) * proporcao_anomalias)
            X_ataques_amostra = resample(
                X_ataque,
                n_samples=min(n_ataques, len(X_ataque)),
                random_state=random_state
            )
            X_treino_if = np.vstack([X_normal, X_ataques_amostra])

        else:
            # Original — treino completo com tudo
            X_treino_if = X_treino_fold

        # Treinar IF
        modelo = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        modelo.fit(X_treino_if)

        # Predição e conversão
        pred_raw = modelo.predict(X_teste_fold)
        y_pred   = np.where(pred_raw == 1, 0, 1)

        # Métricas do fold
        fold_result = {
            "fold"     : fold,
            "acuracia" : accuracy_score(y_teste_fold, y_pred),
            "precisao" : precision_score(y_teste_fold, y_pred, zero_division=0),
            "recall"   : recall_score(y_teste_fold, y_pred, zero_division=0),
            "f1"       : f1_score(y_teste_fold, y_pred, zero_division=0)
        }
        resultados.append(fold_result)

        print(f"Fold {fold}: "
              f"Acurácia={fold_result['acuracia']:.4f} | "
              f"Precisão={fold_result['precisao']:.4f} | "
              f"Recall={fold_result['recall']:.4f} | "
              f"F1={fold_result['f1']:.4f}")

    # Médias e desvios
    print("\n--- Resultado Final ---")
    medias = {}
    desvios = {}
    for metrica in ["acuracia", "precisao", "recall", "f1"]:
        valores = [r[metrica] for r in resultados]
        medias[metrica]  = np.mean(valores)
        desvios[metrica] = np.std(valores)
        print(f"{metrica.capitalize():10}: "
              f"{medias[metrica]:.4f} ± {desvios[metrica]:.4f}")

    # Gráfico
    plotar_resultados_cv(resultados, medias, desvios,
                         variante, nome_dataset)

    return medias, desvios, resultados


# ======================================
# GRÁFICO — MÉTRICAS POR FOLD
# ======================================

def plotar_resultados_cv(resultados, medias, desvios,
                          variante, nome_dataset):

    folds    = [r["fold"]     for r in resultados]
    acuracias = [r["acuracia"] for r in resultados]
    precisoes = [r["precisao"] for r in resultados]
    recalls   = [r["recall"]   for r in resultados]
    f1s       = [r["f1"]       for r in resultados]

    plt.figure(figsize=(10, 5))
    plt.plot(folds, acuracias, marker="o", label="Acurácia",  color="steelblue")
    plt.plot(folds, precisoes, marker="s", label="Precisão",  color="seagreen")
    plt.plot(folds, recalls,   marker="^", label="Recall",    color="tomato")
    plt.plot(folds, f1s,       marker="D", label="F1-score",  color="darkorange")

    # Linha horizontal da média de F1
    plt.axhline(y=medias["f1"], color="darkorange",
                linestyle="--", alpha=0.5,
                label=f"F1 médio: {medias['f1']:.4f} ± {desvios['f1']:.4f}")

    plt.title(f"Validação Cruzada — IF {variante} / {nome_dataset}\n"
              f"(StratifiedKFold, 5 folds)")
    plt.xlabel("Fold")
    plt.ylabel("Valor da Métrica")
    plt.xticks(folds)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("resultados", exist_ok=True)
    nome_arquivo = (f"resultados/cv_if_{variante}_"
                    f"{nome_dataset.lower().replace('-','_')}.png")
    plt.savefig(nome_arquivo, dpi=150)
    plt.show()
    print(f"Figura salva em: {nome_arquivo}")