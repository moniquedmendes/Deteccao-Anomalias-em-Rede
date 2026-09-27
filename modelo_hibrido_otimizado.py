# modelo_hibrido_otimizado.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import os


# ======================================
# FUNCAO PRINCIPAL
# ======================================

def treinar_hibrido_otimizado(X_treino, X_val, X_teste,
                               y_treino, y_val, y_teste,
                               params_rf,
                               variante_if="limpo",
                               contamination_if=0.1,
                               proporcao_anomalias=0.03,
                               nome_dataset="NSL-KDD",
                               experimento="013"):
    """
    Treina o modelo hibrido otimizado combinando:
    - Random Forest com hiperparametros definidos externamente
    - Isolation Forest com variante de treino escolhida

    CORRECAO (ver auditoria): o threshold do IF agora e escolhido
    usando os escores da VALIDACAO (X_val/y_val), nunca do teste.
    O conjunto de teste (X_teste/y_teste) so e usado UMA VEZ, no
    final, com o threshold ja fixado, para gerar a metrica reportada.

    Parametros:
        X_treino, y_treino  : fatia de treino final (ja SEM a validacao)
        X_val, y_val         : fatia de validacao, separada do treino
                                original -- usada so para escolher o
                                threshold
        X_teste, y_teste     : conjunto de teste original do dataset,
                                nunca usado para tomar decisoes
        params_rf           : dicionario com os melhores parametros do RF
        variante_if          : "limpo" (0%), "3pct" (~3%) ou "original"
        contamination_if     : parametro contamination do IF
        proporcao_anomalias : usado na variante "3pct"
        nome_dataset         : nome do dataset
        experimento           : numero do experimento para os arquivos
    """

    print("=" * 60)
    print(f"HIBRIDO OTIMIZADO -- {nome_dataset} -- Experimento {experimento}")
    print(f"RF params: {params_rf}")
    print(f"IF variante: {variante_if} | contamination: {contamination_if}")
    print("=" * 60)

    # ======================================
    # 1. TREINAR RF COM PARAMETROS OTIMIZADOS
    # ======================================
    print("\nTreinando Random Forest otimizado...")
    modelo_rf = RandomForestClassifier(
        **params_rf,
        random_state=42,
        n_jobs=-1
    )
    modelo_rf.fit(X_treino, y_treino)
    print("RF concluido.")

    # ======================================
    # 2. TREINAR IF COM VARIANTE ESCOLHIDA
    # ======================================
    print(f"\nTreinando Isolation Forest ({variante_if})...")

    if variante_if == "limpo":
        X_treino_if = X_treino[y_treino == 0]
        print(f"Registros no treino IF: {len(X_treino_if)} (apenas normais)")

    elif variante_if == "3pct":
        from sklearn.utils import resample
        X_normal = X_treino[y_treino == 0]
        X_ataque = X_treino[y_treino == 1]
        n_ataques = int(len(X_normal) * proporcao_anomalias)
        X_ataques_amostra = resample(
            X_ataque,
            n_samples=min(n_ataques, len(X_ataque)),
            random_state=42
        )
        X_treino_if = np.vstack([X_normal, X_ataques_amostra])
        print(f"Registros no treino IF: {len(X_treino_if)} ({proporcao_anomalias*100:.0f}% anomalias)")

    else:
        X_treino_if = X_treino
        print(f"Registros no treino IF: {len(X_treino_if)} (original)")

    modelo_if = IsolationForest(
        n_estimators=100,
        contamination=contamination_if,
        random_state=42,
        n_jobs=-1
    )
    modelo_if.fit(X_treino_if)
    print("IF concluido.")

    # ======================================
    # 3. ANALISE DE THRESHOLDS -- AGORA NA VALIDACAO
    # ======================================
    print("\nAnalisando thresholds (usando a VALIDACAO, nao o teste)...")
    y_pred_rf_val = modelo_rf.predict(X_val)
    escores_if_val = modelo_if.score_samples(X_val)

    thresholds, resultados_thresh = analisar_thresholds_otimizado(
        y_pred_rf_val, escores_if_val, y_val, nome_dataset, experimento
    )

    # Threshold com melhor F1 -- calculado na validacao
    melhor = max(resultados_thresh, key=lambda x: x["f1"])
    threshold_escolhido = melhor["threshold"]

    print(f"\nThreshold escolhido (validacao): {threshold_escolhido:.4f}")
    print(f"F1 na validacao: {melhor['f1']:.4f}")

    # ======================================
    # 4. APLICACAO FINAL NO TESTE -- UMA UNICA VEZ
    # ======================================
    print(f"\nAplicando threshold ja fixado ({threshold_escolhido:.4f}) ao TESTE...")

    y_pred_rf_teste = modelo_rf.predict(X_teste)
    escores_if_teste = modelo_if.score_samples(X_teste)

    y_pred_hibrido = np.where(
        (y_pred_rf_teste == 1) | (escores_if_teste < threshold_escolhido),
        1, 0
    )

    metricas = {
        "acuracia" : accuracy_score(y_teste, y_pred_hibrido),
        "precisao" : precision_score(y_teste, y_pred_hibrido, zero_division=0),
        "recall"   : recall_score(y_teste, y_pred_hibrido, zero_division=0),
        "f1"       : f1_score(y_teste, y_pred_hibrido, zero_division=0)
    }

    print("\n--- Resultados Finais (conjunto de TESTE, threshold ja fixo) ---")
    print(f"Acuracia  : {metricas['acuracia']:.4f}")
    print(f"Precisao  : {metricas['precisao']:.4f}")
    print(f"Recall    : {metricas['recall']:.4f}")
    print(f"F1-score  : {metricas['f1']:.4f}")
    print("\nRelatorio completo:")
    print(classification_report(y_teste, y_pred_hibrido,
                                 target_names=["Normal", "Ataque"],
                                 labels=[0, 1],
                                 zero_division=0))

    # ======================================
    # 5. GRAFICOS
    # ======================================
    plotar_matriz_confusao(y_teste, y_pred_hibrido, nome_dataset, experimento)

    return y_pred_hibrido, metricas, threshold_escolhido, escores_if_teste, y_pred_rf_teste


# ======================================
# ANALISE DE THRESHOLDS
# ======================================

def analisar_thresholds_otimizado(y_pred_rf, escores_if, y_teste,
                                   nome_dataset, experimento):

    # Grade mais granular baseada nos escores reais.
    # NOTA: se o threshold vencedor cair no extremo p1 (o mais
    # restritivo da grade), isso e sinal de que o ponto otimo real
    # pode estar alem do que foi testado -- vale checar o print de
    # "Threshold escolhido" e, se ele bater em -0,6492/-0,5576-like
    # no limite inferior, rodar de novo com p1 no lugar de p10.
    p1  = np.percentile(escores_if, 1)
    p50 = np.percentile(escores_if, 50)

    thresholds = np.linspace(p1, p50, 20).tolist()

    print(f"\n{'Threshold':>12} | {'Acuracia':>9} | {'Precisao':>9} | "
          f"{'Recall':>9} | {'F1':>9}")
    print("-" * 60)

    resultados = []

    for t in thresholds:
        y_pred = np.where(
            (y_pred_rf == 1) | (escores_if < t),
            1, 0
        )
        r = {
            "threshold": t,
            "acuracia" : accuracy_score(y_teste, y_pred),
            "precisao" : precision_score(y_teste, y_pred, zero_division=0),
            "recall"   : recall_score(y_teste, y_pred, zero_division=0),
            "f1"       : f1_score(y_teste, y_pred, zero_division=0)
        }
        resultados.append(r)
        print(f"{t:>12.4f} | {r['acuracia']:>9.4f} | {r['precisao']:>9.4f} | "
              f"{r['recall']:>9.4f} | {r['f1']:>9.4f}")

    plotar_thresholds(thresholds, resultados, nome_dataset, experimento)

    return thresholds, resultados


# ======================================
# GRAFICO -- THRESHOLDS
# ======================================

def plotar_thresholds(thresholds, resultados, nome_dataset, experimento):

    f1s       = [r["f1"]       for r in resultados]
    recalls   = [r["recall"]   for r in resultados]
    precisoes = [r["precisao"] for r in resultados]

    melhor_f1   = max(f1s)
    melhor_t    = thresholds[f1s.index(melhor_f1)]

    plt.figure(figsize=(11, 5))
    plt.plot(thresholds, f1s,       marker="o", color="darkorange",
             linewidth=2, label="F1-score")
    plt.plot(thresholds, recalls,   marker="s", color="tomato",
             linewidth=2, label="Recall")
    plt.plot(thresholds, precisoes, marker="^", color="steelblue",
             linewidth=2, label="Precisao")
    plt.axvline(x=melhor_t, color="darkorange", linestyle="--",
                alpha=0.6, label=f"Melhor threshold: {melhor_t:.4f}")
    plt.title(f"Analise de Thresholds -- Hibrido Otimizado / {nome_dataset}")
    plt.xlabel("Threshold (escore IF)")
    plt.ylabel("Valor da Metrica")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("resultados", exist_ok=True)
    nome = f"resultados/thresholds_hibrido_{experimento}_{nome_dataset.lower().replace('-','_')}.png"
    plt.savefig(nome, dpi=150)
    plt.show()
    print(f"Figura salva em: {nome}")


# ======================================
# GRAFICO -- MATRIZ DE CONFUSAO
# ======================================

def plotar_matriz_confusao(y_teste, y_pred, nome_dataset, experimento):

    cm = confusion_matrix(y_teste, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Ataque"],
                yticklabels=["Normal", "Ataque"])
    plt.title(f"Matriz de Confusao -- Hibrido Otimizado / {nome_dataset}")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")
    plt.tight_layout()

    os.makedirs("resultados", exist_ok=True)
    nome = f"resultados/matriz_confusao_hibrido_{experimento}_{nome_dataset.lower().replace('-','_')}.png"
    plt.savefig(nome, dpi=150)
    plt.show()
    print(f"Figura salva em: {nome}")