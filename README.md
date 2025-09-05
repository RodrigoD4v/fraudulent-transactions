# Modelo de Detecção de Fraude em Transações Financeiras

Este projeto desenvolve e valida um modelo de Machine Learning para a detecção de fraudes em transações financeiras. A solução utiliza o algoritmo **XGBoost** para identificar padrões fraudulentos em um dataset massivo de mais de 6 milhões de transações, com 91% recall(macro).

---

## Visão Geral do Projeto

O objetivo foi construir um modelo robusto para identificar transações fraudulentas (uma classe minoritária do dataset) de forma precisa e escalável.

* **Metodologia:** Utilizou-se uma amostra estratificada de 500k transações para treinar e comparar seis modelos de classificação. O modelo de melhor desempenho, o **XGBoost**, foi então validado em todo o conjunto de dados raw (6.4 milhões de transações).
* **Resultados:** O modelo alcançou uma performance excepcional, mantendo a métrica **AUC-ROC em 0.99** e a métrica **AUC-PR em 0.80** no conjunto de dados completo, provando sua capacidade de generalização, e eficácia em larga escala.

Para uma análise completa e detalhada das métricas e da metodologia, consulte os relatórios na pasta [reports/](reports/).

---

## Como Usar o Projeto

Siga os passos abaixo para reproduzir os resultados ou usar o modelo treinado.  
Observação: É obrigatório ter Python 3.9.19 instalado para executar os fluxos abaixo.

### Instalação do Make (Opcional)
Se você estiver no Windows, instale o `make` usando o **Git Bash** para aproveitar a automação. Para macOS e Linux, o `make` geralmente já vem pré-instalado.

---

## 1. Obtenção dos Dados

A base de dados utilizada é a [**Fraudulent Transactions Data**](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data) do Kaggle. Para obter os dados, siga as instruções na página da competição e salve os arquivos CSV na pasta `data/raw/`.

Antes de colocar os dados, é necessário criar a estrutura de pastas do projeto. Você pode fazer isso manualmente ou usando o Makefile:

Opção 1: Criar pastas manualmente (Windows CMD/PowerShell)

      md data\raw data\interim data\processed

Opção 2: Criar pastas automaticamente usando o Makefile (Linux/Mac ou Windows com Git Bash + make)

      make setup_dirs

---

## 2. Fluxos de Execução
Existem duas formas de rodar o projeto:  
- **Fluxo Manual**: indicado para usuários de Windows via CMD/PowerShell.  
- **Fluxo Automático via Makefile**: recomendado para Linux/Mac (já incluí make) ou Windows com Git Bash (necessário instalar o utilitário make)

1. **Clone o Repositório:** Abra seu terminal e clone o projeto com o comando `git clone`.

      ```bash
      git clone https://github.com/RodrigoD4v/fraudulent-transactions.git
      ```
      
      ```bash
      cd fraudulent-transactions
      ```
   
      Abra o projeto no seu editor de código favorito:
       
      ```bash
      # Para VS Code
      code .
       
      # Para PyCharm
      charm .
      ```

### 2.1 Fluxo Manual (Windows CMD/PowerShell)  
Para usuários do Windows que preferem não usar `make`:

1 **Crie e ative o ambiente virtual (Python 3.9.19 necessário):** 

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Instale as dependências do projeto com o comando**:

   ```bash
   pip install -r requirements.txt
   ```
   
3. **Instale o pacote local para tornar os scripts da pasta src importáveis**:

   ```bash
   pip install -e .
   ```
   
4. **Execute os Scripts nessa ordem**:

   ```bash
   python src/data/make_dataset.py data/raw/Fraud.csv data/interim/Fraud_sample.parquet --nrows 500000
   python src/features/build_features.py
   python src/models/train_model.py
   python src/models/predict_model.py
   ```

### 2.2 Fluxo via Makefile (Linux/Mac ou Windows Git Bash)
No Windows, use o Git Bash tendo `make` instalado para rodar os comandos abaixo

1. **Configuração inicial (cria a venv, testa a versão do Python e instala as dependências):**

      ```bash
      make requirements
      ```
   
2. **Execute os Scripts nessa ordem**:

      ```bash
      make data        # Processa os dados
      make features    # Gera features
      make train       # Treina o modelo
      make predict     # Faz previsões
      ```

Os gráficos de desempenho são salvos automaticamente na pasta `reports/figures/` após rodar o pipeline (train e predict), independentemente do fluxo que você escolher.

---

## Organização do Projeto

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

---

## Licença e Agradecimentos

Este projeto está sob a licença MIT.

Este projeto foi construído com base no template cookiecutter-data-science.
