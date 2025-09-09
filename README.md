# Desafio Indicium - Predição de Notas IMDB

## Descrição do Projeto

Este projeto foi desenvolvido como parte do desafio técnico para a posição de Cientista de Dados na Indicium. O objetivo é analisar dados cinematográficos do IMDB para auxiliar o estúdio PProductions na tomada de decisões sobre futuras produções.

## Estrutura do Projeto

```
├── data/
│   └── raw/
│       └── desafio_indicium_imdb.csv
├── models/
│   ├── ImdbRatingModel.pkl
│   └── LabelEncoders.pkl
├── notebooks/
│   └── ImdbMovieAnalysis.ipynb
├── src/
│   └── ImdbAnalysis.py
├── requirements.txt
├── run_project.py
└── README.md
```

## Instalação e Execução

### Pré-requisitos
- Python 3.8+
- pip

### Instalação das Dependências

```bash
pip install -r requirements.txt
```

### Executando o Projeto

#### Opção 1: Jupyter Notebook (Recomendado)
```bash
jupyter notebook notebooks/ImdbMovieAnalysis.ipynb
```
#### Opção 2: Script Python
```bash
python app.py
```
Esse script executa todo o pipeline, treina e salva o modelo, e valida o funcionamento do projeto.

## Reprodutibilidade do Modelo

O modelo treinado (`ImdbRatingModel.pkl`) e os encoders (`LabelEncoders.pkl`) podem ser carregados em qualquer ambiente Python para realizar novas predições. Veja exemplo:

```python
import joblib
model = joblib.load('models/ImdbRatingModel.pkl')
encoders = joblib.load('models/LabelEncoders.pkl')
# Use model.predict(features) para novas predições
```

## Dependências

As principais bibliotecas utilizadas são:
- pandas==2.2.2
- numpy==1.26.4
- matplotlib==3.8.4
- seaborn==0.13.2
- scikit-learn==1.4.2
- xgboost==2.0.3
- lightgbm==4.3.0
- plotly==5.22.0
- wordcloud==1.9.3
- textblob==0.18.0.post0
- joblib==1.4.2

## Metodologia

### 1. Análise Exploratória de Dados (EDA)
- Estatísticas descritivas do dataset
- Análise de distribuições e correlações
- Identificação de padrões por gênero e década
- Visualizações interativas com Plotly e Matplotlib

### 2. Engenharia de Features
- Extração de características numéricas (runtime, ano)
- Criação de features derivadas (idade do filme, decade)
- Análise de sentimento das sinopses
- Encoding de variáveis categóricas

### 3. Análise Textual
- Processamento de sinopses com TF-IDF
- Nuvem de palavras
- Análise de sentimento com TextBlob
- Classificação de gêneros a partir do texto

### 4. Modelagem Preditiva
- Múltiplos algoritmos testados:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - Regressão Linear/Ridge/Lasso
- Cross-validation para avaliação robusta
- Seleção do melhor modelo baseado em R²

### 5. Avaliação e Interpretação
- Análise de resíduos
- Importância das features
- Métricas de performance (RMSE, MAE, R²)

## Principais Resultados

### Recomendações de Negócio

1. **Filme Recomendado:** The Godfather (1972) - Nota 9.2
   - Gênero: Crime, Drama
   - Razão: Maior nota IMDB com apelo universal

2. **Fatores para Alto Faturamento:**
   - Número de votos (popularidade)
   - Meta Score (crítica especializada)
   - Gêneros: Action, Adventure, Sci-Fi
   - Certificação adequada ao público-alvo

3. **Insights da Análise Textual:**
   - Sinopses mais longas correlacionam com notas mais altas
   - Sentimento positivo nas descrições é importante
   - Possível inferir gênero através de palavras-chave específicas

### Performance do Modelo

- **Melhor Modelo:** RandomForest
- **R² Score:** 0.4238
- **RMSE:** 0.1945
- **Tipo:** Regressão (predição de valores contínuos)

### Predição para The Shawshank Redemption

- **Predição:** 8.73
- **Real:** 9.3
- **Diferença:** 0.57

## Arquivos Gerados

1. **ImdbRatingModel.pkl:** Modelo treinado para predições
2. **LabelEncoders.pkl:** Encoders para variáveis categóricas
3. **ImdbMovieAnalysis.ipynb:** Notebook completo com análises
4. **ImdbAnalysis.py:** Código modularizado para produção
5. **app.py:** Script para execução automatizada
