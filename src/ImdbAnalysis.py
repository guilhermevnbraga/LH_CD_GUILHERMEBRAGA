import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from wordcloud import WordCloud
from textblob import TextBlob
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
import os
import unicodedata
warnings.filterwarnings('ignore')

class ProcessadorDados:
    def __init__(self, CaminhoDados):
        self.CaminhoDados = CaminhoDados
        self.Dados = None
        self.DadosProcessados = None
        self.Encoders = {}
        self.Normalizador = StandardScaler()

    def CarregarDados(self):
        self.Dados = pd.read_csv(self.CaminhoDados)
        return self.Dados

    def AnalisarDados(self):
        Informacoes = {
            'Dimensao': self.Dados.shape,
            'Colunas': list(self.Dados.columns),
            'Tipos': dict(self.Dados.dtypes),
            'Nulos': dict(self.Dados.isnull().sum()),
            'Duplicados': self.Dados.duplicated().sum(),
            'Resumo': self.Dados.describe()
        }
        return Informacoes

    def LimparDados(self):
        DadosLimpos = self.Dados.copy()
        DadosLimpos = DadosLimpos.dropna(subset=['IMDB_Rating'])
        if 'Runtime' in DadosLimpos.columns:
            DadosLimpos['RuntimeMinutes'] = DadosLimpos['Runtime'].str.extract('(\d+)').astype(float)
            DadosLimpos = DadosLimpos.drop('Runtime', axis=1)
        if 'Gross' in DadosLimpos.columns:
            DadosLimpos['GrossRevenue'] = DadosLimpos['Gross'].str.replace(',', '').astype(float)
            DadosLimpos = DadosLimpos.drop('Gross', axis=1)
        ColunasCateg = ['Certificate', 'Genre', 'Director']
        for Coluna in ColunasCateg:
            if Coluna in DadosLimpos.columns:
                DadosLimpos[Coluna] = DadosLimpos[Coluna].fillna('Unknown')
        ColunasNum = ['Meta_score', 'No_of_Votes', 'GrossRevenue', 'RuntimeMinutes']
        for Coluna in ColunasNum:
            if Coluna in DadosLimpos.columns:
                DadosLimpos[Coluna] = DadosLimpos[Coluna].fillna(DadosLimpos[Coluna].median())
        if 'Released_Year' in DadosLimpos.columns:
            DadosLimpos['Released_Year'] = pd.to_numeric(DadosLimpos['Released_Year'], errors='coerce')
            DadosLimpos['Released_Year'] = DadosLimpos['Released_Year'].fillna(DadosLimpos['Released_Year'].median())
        self.DadosProcessados = DadosLimpos
        return self.DadosProcessados

    def CriarFeatures(self):
        if self.DadosProcessados is None:
            self.LimparDados()
        Dados = self.DadosProcessados.copy()
        if 'Overview' in Dados.columns:
            Dados['OverviewLength'] = Dados['Overview'].str.len().fillna(0)
            Dados['OverviewSentiment'] = Dados['Overview'].fillna('').apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        if 'Genre' in Dados.columns:
            Generos = Dados['Genre'].str.get_dummies(sep=', ')
            Dados = pd.concat([Dados, Generos], axis=1)
        if 'Released_Year' in Dados.columns:
            Dados['MovieAge'] = 2024 - Dados['Released_Year']
            Dados['DecadeReleased'] = (Dados['Released_Year'] // 10) * 10
        if 'No_of_Votes' in Dados.columns:
            Dados['VotesLog'] = np.log1p(Dados['No_of_Votes'])
        if 'GrossRevenue' in Dados.columns:
            Dados['RevenueLog'] = np.log1p(Dados['GrossRevenue'])
        
        def limpar_nome_coluna(col):
            col = unicodedata.normalize('NFKD', col).encode('ASCII', 'ignore').decode('ASCII')
            col = col.replace(' ', '_').replace('-', '_')
            return ''.join(c for c in col if c.isalnum() or c == '_')
        Dados.columns = [limpar_nome_coluna(col) for col in Dados.columns]
        self.DadosProcessados = Dados
        return self.DadosProcessados

    def PrepararModelagem(self, ColunaAlvo='IMDB_Rating'):
        if self.DadosProcessados is None:
            self.CriarFeatures()
        Dados = self.DadosProcessados.copy()
        ColunasRemover = ['Series_Title', 'Overview', 'Star1', 'Star2', 'Star3', 'Star4']
        for Coluna in ColunasRemover:
            if Coluna in Dados.columns:
                Dados = Dados.drop(Coluna, axis=1)
        ColunasCateg = ['Certificate', 'Director', 'Genre']
        for Coluna in ColunasCateg:
            if Coluna in Dados.columns:
                if Dados[Coluna].nunique() > 50:
                    Principais = Dados[Coluna].value_counts().head(20).index
                    Dados[Coluna] = Dados[Coluna].apply(lambda x: x if x in Principais else 'Other')
                self.Encoders[Coluna] = LabelEncoder()
                Dados[Coluna] = self.Encoders[Coluna].fit_transform(Dados[Coluna])
        X = Dados.drop(ColunaAlvo, axis=1)
        y = Dados[ColunaAlvo]
        return X, y

class VisualizadorDados:
    def __init__(self, Dados):
        self.Dados = Dados

    def GraficosBasicos(self):
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        self.Dados['IMDB_Rating'].hist(bins=30, ax=axes[0,0], alpha=0.7)
        axes[0,0].set_title('Distribuição IMDB Rating')
        if 'Released_Year' in self.Dados.columns:
            self.Dados['Released_Year'].hist(bins=30, ax=axes[0,1], alpha=0.7)
            axes[0,1].set_title('Distribuição Ano de Lançamento')
        if 'No_of_Votes' in self.Dados.columns:
            axes[1,0].scatter(self.Dados['No_of_Votes'], self.Dados['IMDB_Rating'], alpha=0.5)
            axes[1,0].set_title('Votos vs Nota')
        if 'Meta_score' in self.Dados.columns:
            Validos = self.Dados.dropna(subset=['Meta_score'])
            axes[1,1].scatter(Validos['Meta_score'], Validos['IMDB_Rating'], alpha=0.5)
            axes[1,1].set_title('Meta Score vs IMDB Rating')
        plt.tight_layout()
        plt.show()

    def AnaliseGeneros(self):
        if 'Genre' not in self.Dados.columns:
            return
        Contagem = self.Dados['Genre'].str.split(', ').explode().value_counts().head(15)
        plt.figure(figsize=(12, 8))
        Contagem.plot(kind='bar')
        plt.title('Top Gêneros')
        plt.show()

    def NuvemPalavras(self):
        if 'Overview' not in self.Dados.columns:
            return
        Texto = ' '.join(self.Dados['Overview'].dropna().astype(str))
        Nuvem = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(Texto)
        plt.figure(figsize=(12, 6))
        plt.imshow(Nuvem, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    def MatrizCorrelacao(self):
        Numericos = self.Dados.select_dtypes(include=[np.number])
        plt.figure(figsize=(12, 10))
        sns.heatmap(Numericos.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação')
        plt.show()

class TreinadorModelo:
    def __init__(self):
        self.Modelos = {}
        self.MelhorModelo = None
        self.MelhorScore = -np.inf

    def TreinarMultiplos(self, X_train, X_test, y_train, y_test):
        Modelos = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1)
        }
        Resultados = {}
        for Nome, Modelo in Modelos.items():
            Modelo.fit(X_train, y_train)
            Prev = Modelo.predict(X_test)
            Rmse = np.sqrt(mean_squared_error(y_test, Prev))
            R2 = r2_score(y_test, Prev)
            Mae = mean_absolute_error(y_test, Prev)
            Resultados[Nome] = {'Modelo': Modelo, 'RMSE': Rmse, 'R2': R2, 'MAE': Mae}
            if R2 > self.MelhorScore:
                self.MelhorScore = R2
                self.MelhorModelo = Modelo
                self.NomeMelhor = Nome
        self.Modelos = Resultados
        return Resultados

    def Otimizar(self, X_train, y_train):
        if self.NomeMelhor == 'RandomForest':
            Parametros = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
            Busca = GridSearchCV(RandomForestRegressor(random_state=42), Parametros, cv=5, scoring='r2')
        elif self.NomeMelhor == 'XGBoost':
            Parametros = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
            Busca = GridSearchCV(xgb.XGBRegressor(random_state=42), Parametros, cv=5, scoring='r2')
        else:
            return self.MelhorModelo
        Busca.fit(X_train, y_train)
        self.MelhorModelo = Busca.best_estimator_
        return self.MelhorModelo

    def Salvar(self, Caminho):
        os.makedirs(os.path.dirname(Caminho), exist_ok=True)
        joblib.dump(self.MelhorModelo, Caminho)

class SistemaRecomendacao:
    def __init__(self, Dados, Modelo):
        self.Dados = Dados
        self.Modelo = Modelo

    def Recomendar(self):
        Top = self.Dados.nlargest(10, 'IMDB_Rating')
        Filme = Top.iloc[0]
        return {'Titulo': Filme.get('Series_Title', 'Desconhecido'), 'Nota': Filme['IMDB_Rating'], 'Ano': Filme.get('Released_Year', 'Desconhecido'), 'Genero': Filme.get('Genre', 'Desconhecido')}

    def FatoresFaturamento(self):
        if 'GrossRevenue' not in self.Dados.columns:
            return {}
        DadosValidos = self.Dados.dropna(subset=['GrossRevenue'])
        if len(DadosValidos) == 0:
            return {}
        Correlacao = DadosValidos.select_dtypes(include=[np.number]).corrwith(DadosValidos['GrossRevenue'])
        Fatores = Correlacao.abs().sort_values(ascending=False).head(10)
        return dict(Fatores)

    def PreverNota(self, Features):
        try:
            return float(self.Modelo.predict([Features])[0])
        except:
            return 7.5

def ExecutarAnalise():
    RaizProjeto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Caminho = os.path.join(RaizProjeto, 'data', 'raw', 'desafio_indicium_imdb.csv')
    Proc = ProcessadorDados(Caminho)
    Dados = Proc.CarregarDados()
    Proc.LimparDados()
    Proc.CriarFeatures()
    X, y = Proc.PrepararModelagem()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    Treinador = TreinadorModelo()
    Resultados = Treinador.TreinarMultiplos(X_train, X_test, y_train, y_test)
    Treinador.Otimizar(X_train, y_train)
    Treinador.Salvar('../models/ImdbRatingModel.pkl')
    Sistema = SistemaRecomendacao(Proc.DadosProcessados, Treinador.MelhorModelo)
    return Proc, Treinador, Sistema

if __name__ == "__main__":
    ExecutarAnalise()
