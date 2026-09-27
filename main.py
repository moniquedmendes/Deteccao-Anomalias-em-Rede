# main.py


import numpy as np
from sklearn.model_selection import train_test_split

from preprocessamento import carregar_e_preparar
from preprocessamento_unsw import carregar_e_preparar_unsw

from random_forest import treinar_random_forest
from isolation_forest import treinar_isolation_forest
from isolation_forest_variantes import (
    treinar_if_treino_limpo,
    treinar_if_3pct_anomalias,
)
from modelo_hibrido_otimizado import treinar_hibrido_otimizado
from validacao_cruzada_if import validacao_cruzada_if
from visualizacao_fronteira import plotar_fronteira_2d


# ==========================================================
# Melhores hiperparametros do RF encontrados na etapa de
# otimizacao. AJUSTE AQUI se voce tiver outros valores vindos
# de um grid search / random search separado.
# ==========================================================
PARAMS_RF = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}


def rodar_experimentos(nome_dataset, X_treino, X_teste, y_treino, y_teste,
                        feature_names, experimento_hibrido="013",
                        rodar_cv=True):

    print("\n" + "#" * 60)
    print(f"# DATASET: {nome_dataset}")
    print("#" * 60)

    # ------------------------------------------------------
    # 1. Random Forest
    # ------------------------------------------------------
    modelo_rf, y_pred_rf, metricas_rf = treinar_random_forest(
        X_treino, X_teste, y_treino, y_teste,
        feature_names=feature_names,
        nome_dataset=nome_dataset
    )

    # ------------------------------------------------------
    # 2. Isolation Forest - treino "puro" (todo o treino, sem
    #    filtrar nem amostrar nada, so pra ter esse baseline)
    # ------------------------------------------------------
    modelo_if, y_pred_if, escores_if, metricas_if = treinar_isolation_forest(
        X_treino, X_teste, y_teste, nome_dataset=nome_dataset
    )

    # ------------------------------------------------------
    # 3. Isolation Forest - treino limpo (0% anomalias)
    # ------------------------------------------------------
    modelo_if_limpo, y_pred_if_limpo, escores_if_limpo, metricas_if_limpo = \
        treinar_if_treino_limpo(
            X_treino, X_teste, y_treino, y_teste,
            nome_dataset=nome_dataset
        )

    # ------------------------------------------------------
    # 4. Isolation Forest - treino com ~3% de anomalias
    # ------------------------------------------------------
    modelo_if_3pct, y_pred_if_3pct, escores_if_3pct, metricas_if_3pct = \
        treinar_if_3pct_anomalias(
            X_treino, X_teste, y_treino, y_teste,
            nome_dataset=nome_dataset
        )

    # ------------------------------------------------------
    # 5. Validacao Cruzada do IF (limpo e 3pct)
    #    Usa o dataset inteiro (treino + teste), ja que o KFold
    #    faz os proprios splits internamente.
    # ------------------------------------------------------
    resultados_cv = {}
    if rodar_cv:
        X_completo = np.vstack([X_treino, X_teste])
        y_completo = np.concatenate([y_treino, y_teste])

        for variante in ["limpo", "3pct"]:
            print(f"\n>>> Validacao Cruzada -- IF variante: {variante}")
            medias, desvios, _ = validacao_cruzada_if(
                X_completo, y_completo,
                variante=variante,
                nome_dataset=nome_dataset
            )
            resultados_cv[variante] = {"medias": medias, "desvios": desvios}

    # ------------------------------------------------------
    # 6. Modelo Hibrido Otimizado
    #    separa uma fatia de VALIDACAO do treino para escolher
    #    o threshold sem tocar no teste; roda para as 3 variantes
    #    de IF ("limpo", "3pct", "original") pra comparar.
    # ------------------------------------------------------
    X_treino_final, X_val, y_treino_final, y_val = train_test_split(
        X_treino, y_treino,
        test_size=0.2,
        random_state=42,
        stratify=y_treino
    )

    resultados_hibrido = {}
    predicoes_hibrido = {}
    for variante in ["limpo", "3pct", "original"]:
        print(f"\n>>> Hibrido Otimizado -- variante IF: {variante}")
        y_pred_h, metricas_h, threshold_h, _, _ = treinar_hibrido_otimizado(
            X_treino_final, X_val, X_teste,
            y_treino_final, y_val, y_teste,
            params_rf=PARAMS_RF,
            variante_if=variante,
            nome_dataset=nome_dataset,
            experimento=f"{experimento_hibrido}_{variante}"
        )
        resultados_hibrido[variante] = metricas_h
        predicoes_hibrido[variante] = y_pred_h

    # ------------------------------------------------------
    # 7. Visualizacao de fronteira 2D (PCA)
    #    Uma para o RF puro, outra para o melhor hibrido
    #    (maior F1 entre as variantes testadas acima).
    # ------------------------------------------------------
    plotar_fronteira_2d(X_teste, y_teste, y_pred_rf,
                         nome_modelo="Random Forest",
                         nome_dataset=nome_dataset)

    melhor_variante = max(resultados_hibrido, key=lambda v: resultados_hibrido[v]["f1"])
    plotar_fronteira_2d(X_teste, y_teste, predicoes_hibrido[melhor_variante],
                         nome_modelo=f"Hibrido Otimizado ({melhor_variante})",
                         nome_dataset=nome_dataset)

    return {
        "rf": metricas_rf,
        "if_puro": metricas_if,
        "if_limpo": metricas_if_limpo,
        "if_3pct": metricas_if_3pct,
        "cv": resultados_cv,
        "hibrido": resultados_hibrido,
    }


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("EXPERIMENTOS -- NSL-KDD")
    print("=" * 60)
    X_treino, X_teste, y_treino, y_teste, feature_names = carregar_e_preparar()
    resultados_nslkdd = rodar_experimentos(
        "NSL-KDD", X_treino, X_teste, y_treino, y_teste, feature_names
    )

    print("\n" + "=" * 60)
    print("EXPERIMENTOS -- UNSW-NB15")
    print("=" * 60)
    X_treino_u, X_teste_u, y_treino_u, y_teste_u, feature_names_u = \
        carregar_e_preparar_unsw()
    resultados_unsw = rodar_experimentos(
        "UNSW-NB15", X_treino_u, X_teste_u, y_treino_u, y_teste_u,
        feature_names_u
    )

    # ------------------------------------------------------
    # Resumo final - so pra ter uma visao rapida no terminal
    # ------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESUMO FINAL (F1-score)")
    print("=" * 60)
    for nome, resultados in [("NSL-KDD", resultados_nslkdd),
                              ("UNSW-NB15", resultados_unsw)]:
        print(f"\n--- {nome} ---")
        print(f"RF                 : {resultados['rf']['f1']:.4f}")
        print(f"IF puro            : {resultados['if_puro']['f1']:.4f}")
        print(f"IF limpo (0%)      : {resultados['if_limpo']['f1']:.4f}")
        print(f"IF ~3% anomalias   : {resultados['if_3pct']['f1']:.4f}")
        for variante, cv in resultados["cv"].items():
            print(f"CV IF ({variante:<6})     : "
                  f"{cv['medias']['f1']:.4f} +/- {cv['desvios']['f1']:.4f}")
        for variante, m in resultados["hibrido"].items():
            print(f"Hibrido ({variante:<8}) : {m['f1']:.4f}")