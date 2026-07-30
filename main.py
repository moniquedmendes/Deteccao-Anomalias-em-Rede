# main.py

from preprocessamento import carregar_e_preparar
from random_forest import treinar_random_forest
from isolation_forest import treinar_isolation_forest
from modelo_hibrido import treinar_modelo_hibrido, analisar_thresholds

#aviso: se quiser gerar os modelos de RF ou IF a parte é só comentar a parte do codigo que não é o foco
#porem o modelo hibrido deixa tudo que ta nesse codigo pq ele só junta os dois!!!

# 1. Preparar dados
X_treino, X_teste, y_treino, y_teste, feature_names = carregar_e_preparar()

# 2. Treinar e avaliar Random Forest
modelo_rf, y_pred_rf, metricas_rf = treinar_random_forest(
    X_treino, X_teste, y_treino, y_teste,
    feature_names=feature_names
)

# 3. Isolation Forest
# Atenção: y_treino NÃO é passado  o IF treina sem rótulos
modelo_if, y_pred_if, escores_if, metricas_if = treinar_isolation_forest(
    X_treino, X_teste, y_teste
)

# 4. Análise de thresholds  roda antes para escolher o melhor valor
resultados_threshold = analisar_thresholds(y_pred_rf, escores_if, y_teste)

# 5. Modelo Híbrido  usar o threshold escolhido na análise
y_pred_hibrido, metricas_hibrido = treinar_modelo_hibrido(
    y_pred_rf, escores_if, y_teste,
    threshold=-0.45   # ajustar conforme análise acima
)

#Separar os dois modelos, talvez de para rodar TUDO de uma vez o que não faz tanto sentido
#já que estamos fazendo testes e relatorios separados, acho que assim fica mais facil
#para analizar