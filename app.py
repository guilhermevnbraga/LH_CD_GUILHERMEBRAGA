#!/usr/bin/env python
import os
import sys

def ExecutarProjeto():
    print("=== DESAFIO INDICIUM - PREDIÇÃO NOTAS IMDB ===")
    print("Autor: Guilherme Vinícius Nigro Braga")
    print("="*50)
    
    RaizProjeto = os.path.dirname(os.path.abspath(__file__))
    
    CaminhoDados = os.path.join(RaizProjeto, 'data', 'raw', 'desafio_indicium_imdb.csv')
    CaminhoNotebook = os.path.join(RaizProjeto, 'notebooks', 'ImdbMovieAnalysis.ipynb')
    
    if not os.path.exists(CaminhoDados):
        print(f"ERRO: Dataset não encontrado em {CaminhoDados}")
        return False
    
    if not os.path.exists(CaminhoNotebook):
        print(f"ERRO: Notebook não encontrado em {CaminhoNotebook}")
        return False
    
    print(f"Dataset encontrado: {CaminhoDados}")
    print(f"Notebook encontrado: {CaminhoNotebook}")
    
    print("\nPara rodar a análise completa:")
    print(f"1. Abrir no Jupyter: jupyter notebook {CaminhoNotebook}")
    
    try:
        from src.ImdbAnalysis import ExecutarAnalise
        print("\n=== EXECUTANDO ANÁLISE ===")
        Processador, Treinador, SistemaRecomendacao = ExecutarAnalise()
        return True
    except Exception as e:
        print(f"Erro ao executar análise: {e}")
        return False

if __name__ == "__main__":
    Sucesso = ExecutarProjeto()
    if Sucesso:
        print("\n=== PROJETO CONCLUÍDO COM SUCESSO ===")
    else:
        print("\n=== PROJETO FALHOU ===")
        sys.exit(1)
