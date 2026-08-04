# Detecção de Anomalias em Tráfego de Rede com Machine Learning

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

## Estrutura do Projeto

| Arquivo | Descrição |
|---|---|
| `preprocessamento.py` | Pré-processamento do NSL-KDD |
| `preprocessamento_unsw.py` | Pré-processamento do UNSW-NB15 |
| `random_forest.py` | Modelo supervisionado |
| `isolation_forest.py` | Modelo não supervisionado |
| `modelo_hibrido.py` | Combinação RF + IF |
| `main.py` | Orquestra a execução |
