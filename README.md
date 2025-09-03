# Modelo de Detecção de Fraude em Transações Financeiras

Este projeto desenvolve e valida um modelo de Machine Learning para a detecção de fraudes em transações financeiras. A solução utiliza o algoritmo **XGBoost** para identificar padrões fraudulentos em um dataset massivo de mais de 6 milhões de transações, com 99% de precisão e recall.

---

## Visão Geral do Projeto

O objetivo foi construir um modelo robusto para identificar transações fraudulentas (uma classe minoritária) de forma precisa e escalável.

* **Metodologia:** Utilizou-se uma amostra estratificada de 500k transações para treinar e comparar seis modelos de classificação. O modelo de melhor desempenho, o **XGBoost**, foi então validado em todo o conjunto de dados raw (6.4 milhões de transações).
* **Resultados:** O modelo alcançou uma performance excepcional, mantendo a métrica **AUC-ROC em 0.99** e a métrica **AUC-PR em 0.80** no conjunto de dados completo, provando sua capacidade de generalização e eficácia em larga escala.

Para uma análise completa e detalhada das métricas e da metodologia, consulte os relatórios na pasta [reports/](reports/).

---

## Como Usar o Projeto

Siga os passos abaixo para reproduzir os resultados ou usar o modelo treinado.

### 1. Obtenção dos Dados

A base de dados utilizada é a [**Fraudulent Transactions Data**](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data) do Kaggle. Para obter os dados, siga as instruções na página da competição e salve os arquivos CSV na pasta `data/raw/`.

### 2. Pré-requisitos

Certifique-se de ter o Python 3.9+ instalado. Instale as dependências do projeto com o comando:

```bash
pip install -r requirements.txt
```

## Estrutura

O projeto segue a estrutura padrão cookiecutter-data-science. Os scripts principais estão localizados na pasta src/models/.

## Etapas

1. Processamento de Dados: Rode o script para processar os dados brutos e gerar os arquivos intermediários e processados.

   ```bash
   python src/data/make_dataset.py data/raw/Fraud.csv data/interim/Fraud_sample.parquet --nrows 500000
   ```
   
2. Engenharia de Features: Execute o script para transformar os dados processados em features prontas para o modelo.

   ```bash
   python src/features/build_features.py
   ```
   
3. Treine o Modelo: Rode o script de treinamento para treinar o modelo em uma amostra dos dados.

    ```bash
    python src/models/train_model.py
    ```
    
4. Gere as Previsões: Utilize o modelo treinado para gerar previsões em todo o dataset.

   ```bash
    python src/models/predict_model.py
    ```

5. Visualize os Resultados: Os gráficos de desempenho são salvos automaticamente na pasta **reports/figures/** após rodar o train_model.py e predict_model.py

## Organização do Projeto
------------


    ├── LICENSE
    ├── Makefile            <- Arquivo com comandos para o pipeline (`make data`, `make train`).
    ├── README.md           <- Este arquivo principal do projeto.
    ├── data
    │   ├── external        <- Dados de fontes de terceiros.
    │   ├── interim         <- Dados intermediários que foram transformados.
    │   ├── processed       <- Conjuntos de dados finais e limpos para modelagem.
    │   └── raw             <- Os dados originais e imutáveis.
    │
    ├── docs                <- Documentação do projeto (planos, dicionários de dados).
    │
    ├── models              <- Modelos treinados e serializados, e suas previsões.
    │
    ├── notebooks           <- Jupyter notebooks para análise e exploração dos dados.
    │
    ├── references          <- Dicionários de dados, manuais, e outros materiais de referência.
    │
    ├── reports             <- Análises geradas em HTML, PDF, LaTeX, etc.
    │   └── figures         <- Gráficos e figuras geradas para relatórios.
    │
    ├── requirements.txt    <- Arquivo com as dependências para reprodução do ambiente.
    │
    ├── setup.py            <- Torna o `src` um pacote Python importável.
    ├── src                 <- Código-fonte para o projeto.
    │   ├── __init__.py     <- Torna `src` um módulo Python.
    │   ├── data            <- Scripts para baixar ou gerar dados.
    │   │   └── make_dataset.py
    │   ├── features        <- Scripts para transformar dados brutos em features.
    │   │   └── build_features.py
    │   ├── models          <- Scripts para treinar e usar modelos.
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   └── visualization   <- Scripts para criar visualizações.
    │       └── visualize.py
    │
    └── tox.ini             <- Arquivo de configuração para rodar testes.



--------
## Licença e Agradecimentos

Este projeto está sob a licença MIT.

Este projeto foi construído com base no template cookiecutter-data-science.
