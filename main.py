# main.py

from preprocessamento import carregar_e_preparar
from random_forest import treinar_random_forest

# 1. Preparar dados
X_treino, X_teste, y_treino, y_teste, feature_names = carregar_e_preparar()

# 2. Treinar e avaliar Random Forest
modelo_rf, y_pred_rf, metricas_rf = treinar_random_forest(
    X_treino, X_teste, y_treino, y_teste,
    feature_names=feature_names
)