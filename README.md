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
├── AnaliseCompleta.pdf
├── app.py
├── requirements.txt
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

## Respostas às Perguntas de Negócio

### 1. Qual filme você recomendaria para uma pessoa que você não conhece?
Após analisar as avaliações dos filmes, "The Godfather (1972)" se destaca como o filme com maior nota IMDB. Por isso, ele é recomendado para qualquer público, sendo uma escolha segura.
> **Insight:** Filmes com alta nota tendem a ser clássicos reconhecidos, reforçando a importância de considerar avaliações agregadas na recomendação.

### 2. Quais são os principais fatores que estão relacionados com alta expectativa de faturamento de um filme?
A análise de correlação mostra que o número de votos (popularidade), Meta Score (crítica especializada), gêneros como Action, Adventure, Sci-Fi e certificação adequada ao público-alvo são os principais fatores associados à alta bilheteira.
> **Insight:** Investir em gêneros populares e buscar boas avaliações da crítica pode aumentar a expectativa de receita.

### 3. Quais insights podem ser tirados com a coluna Overview? É possível inferir o gênero do filme a partir dessa coluna?
A análise textual revela que sinopses mais longas estão associadas a notas mais altas e que o sentimento positivo nas descrições é relevante. Além disso, palavras-chave presentes na Overview permitem inferir o gênero do filme, como mostrado na análise de frequência de termos.
> **Insight:** O texto da sinopse é bom meio para entender o perfil do filme e pode ser usado para classificação de gênero e análise de sentimento.

### 4. Como foi feita a previsão da nota do IMDB? Quais variáveis e modelo foram utilizados?
A previsão da nota IMDB foi realizada utilizando variáveis numéricas, categóricas e derivadas, como idade do filme, sentimento da sinopse, log de votos e receita. O problema é de regressão, pois a nota é contínua. Foram testados os modelos random forest, gradient boosting, XGBoost, LightGBM e regressão linear/ridge/lasso. O modelo Random Forest apresentou o melhor desempenho (maior R² e estabilidade na validação cruzada). A métrica escolhida foi o R², pois mede o quanto da variabilidade dos dados é explicada pelo modelo.
> **Insight:** A combinação de diferentes tipos de variáveis e o uso de modelos como o Random Forest são ótimos para entender a complexidade dos dados de filmes.

### 5. Qual seria a nota do IMDB para o filme exemplo fornecido?
Para "The Shawshank Redemption", o modelo previu nota 8.73, enquanto a nota real é 9.3. Isso mostra que o modelo é capaz de se aproximar da avaliação real, mas ainda pode haver diferenças devido a fatores não capturados nos dados.
> **Insight:** A predição próxima da nota real reforça a utilidade do modelo para estimar avaliações de novos filmes, auxiliando decisões do estúdio.
