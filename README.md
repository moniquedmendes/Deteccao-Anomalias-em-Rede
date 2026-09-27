# Detecção de Anomalias em Tráfego de Rede com Machine Learning

Trabalho de Conclusão de Curso que compara abordagens supervisionadas,
não supervisionadas e híbridas para detectar tráfego de rede anômalo
(ataques), usando os datasets **NSL-KDD** e **UNSW-NB15**.

São avaliados três modelos:

- **Random Forest** (supervisionado) — aprende diretamente com os rótulos
  de ataque/normal.
- **Isolation Forest** (não supervisionado) — aprende o padrão do tráfego
  normal e sinaliza desvios como anomalia. Testado em três variantes de
  treino: *puro* (dados originais, sem filtro), *limpo* (0% de anomalias
  no treino) e *~3% anomalias* (cenário mais realista de produção).
- **Modelo Híbrido Otimizado** — combina RF + IF. O threshold do IF é
  escolhido usando um conjunto de **validação** separado do treino, e só
  é aplicado ao teste uma única vez no final, evitando vazamento de dado.

## Datasets

Os datasets não estão incluídos neste repositório por questões de tamanho.

**NSL-KDD:**
Baixar os arquivos `KDDTrain+.txt` e `KDDTest+.txt` e colocar na pasta `data/`.
Disponível em: https://www.unb.ca/cic/datasets/nsl.html

**UNSW-NB15:**
Baixar os arquivos `UNSW_NB15_training-set.csv` e `UNSW_NB15_testing-set.csv`
e colocar na pasta `data/`.
Disponível em: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

## Como executar

```bash
pip install -r requirements.txt
python main.py
```

O script roda automaticamente todos os experimentos (RF, IF em suas
variantes, validação cruzada do IF, híbrido otimizado e visualizações)
para os dois datasets, em sequência. Os gráficos e figuras gerados são
salvos na pasta `resultados/`, criada automaticamente na primeira execução.

Caso queira rodar só uma parte específica (por exemplo, só o RF), basta
importar e chamar a função correspondente diretamente, sem passar pelo
`main.py`.

## Estrutura do Projeto

| Arquivo | Descrição |
|---|---|
| `preprocessamento.py` | Pré-processamento do NSL-KDD |
| `preprocessamento_unsw.py` | Pré-processamento do UNSW-NB15 |
| `random_forest.py` | Modelo supervisionado (Random Forest) |
| `isolation_forest.py` | Modelo não supervisionado base (Isolation Forest) |
| `isolation_forest_variantes.py` | Variantes de treino do IF (limpo / ~3% anomalias) |
| `validacao_cruzada_if.py` | Validação cruzada estratificada (StratifiedKFold) para o IF |
| `modelo_hibrido_otimizado.py` | Combinação RF + IF, com threshold escolhido na validação |
| `visualizacao_fronteira.py` | Visualização 2D (PCA) das classificações e erros do modelo |
| `main.py` | Orquestra a execução de todos os experimentos |

## Métricas reportadas

Para cada modelo/variante: acurácia, precisão, recall e F1-score, além de
matriz de confusão e (quando aplicável) análise de threshold e gráfico de
fronteira de decisão em 2D.