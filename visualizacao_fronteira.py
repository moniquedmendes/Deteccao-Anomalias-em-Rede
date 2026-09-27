# visualizacao_fronteira.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plotar_fronteira_2d(X_teste, y_teste, y_pred, 
                         nome_modelo="Modelo", 
                         nome_dataset="NSL-KDD"):
    """
    Reduz os dados para 2 dimensões com PCA e plota:
    - Pontos coloridos por classe REAL (normal/ataque)
    - Marcador diferente para erros do modelo (FP e FN)
    """

    print(f"Gerando visualização 2D — {nome_modelo} / {nome_dataset}...")

    # ======================================
    # 1. REDUZIR PARA 2 DIMENSÕES
    # ======================================
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_teste)

    variancia = pca.explained_variance_ratio_
    print(f"Variância explicada pelos 2 componentes: "
          f"{variancia[0]:.1%} + {variancia[1]:.1%} = "
          f"{sum(variancia):.1%}")

    # ======================================
    # 2. SEPARAR OS GRUPOS
    # ======================================
    y_teste_arr = np.array(y_teste)
    y_pred_arr  = np.array(y_pred)

    # Acertos
    vp = (y_teste_arr == 1) & (y_pred_arr == 1)  # ataque correto
    vn = (y_teste_arr == 0) & (y_pred_arr == 0)  # normal correto

    # Erros
    fp = (y_teste_arr == 0) & (y_pred_arr == 1)  # normal previsto como ataque
    fn = (y_teste_arr == 1) & (y_pred_arr == 0)  # ataque previsto como normal

    # ======================================
    # 3. PLOTAR
    # ======================================
    fig, ax = plt.subplots(figsize=(10, 7))

    # Acertos — plotados com transparência para não poluir
    ax.scatter(X_2d[vn, 0], X_2d[vn, 1],
               c="steelblue", marker="x", alpha=0.3, s=15,
               label=f"Normal correto (VN) — {sum(vn)}")

    ax.scatter(X_2d[vp, 0], X_2d[vp, 1],
               c="tomato", marker="x", alpha=0.3, s=15,
               label=f"Ataque correto (VP) — {sum(vp)}")

    # Erros — destacados com marcador maior
    if sum(fp) > 0:
        ax.scatter(X_2d[fp, 0], X_2d[fp, 1],
                   c="orange", marker="^", s=60, edgecolors="black",
                   linewidths=0.5, label=f"Falso Positivo (FP) — {sum(fp)}")

    if sum(fn) > 0:
        ax.scatter(X_2d[fn, 0], X_2d[fn, 1],
                   c="darkred", marker="v", s=60, edgecolors="black",
                   linewidths=0.5, label=f"Falso Negativo (FN) — {sum(fn)}")

    # ======================================
    # 4. FORMATAÇÃO
    # ======================================
    var_total = sum(variancia)
    ax.set_title(
        f"Distribuição das Classificações — {nome_modelo} / {nome_dataset}\n"
        f"(PCA — variância explicada: {var_total:.1%})",
        fontsize=13
    )
    ax.set_xlabel(f"Componente Principal 1 ({variancia[0]:.1%} da variância)")
    ax.set_ylabel(f"Componente Principal 2 ({variancia[1]:.1%} da variância)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()

    import os
    os.makedirs("resultados", exist_ok=True)
    nome_arquivo = f"resultados/fronteira_2d_{nome_modelo.lower().replace(' ', '_')}_{nome_dataset.lower()}.png"
    plt.savefig(nome_arquivo, dpi=150)
    plt.show()
    print(f"Figura salva em: {nome_arquivo}")