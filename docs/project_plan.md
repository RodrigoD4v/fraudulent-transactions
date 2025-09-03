# Plano do Projeto: Modelo de Detecção de Fraude

## 1. Problema de Negócio

O objetivo principal deste projeto é desenvolver um modelo robusto de aprendizado de máquina capaz de identificar transações financeiras fraudulentas. A detecção de fraude é um desafio crítico para instituições financeiras devido ao alto volume de transações e ao desbalanceamento de classes, onde a maioria das transações é legítima. A solução busca minimizar as perdas financeiras causadas por fraudes e otimizar o tempo gasto na análise manual de transações.

## 2. Metodologia

A metodologia adotada segue um fluxo de trabalho padrão de ciência de dados, com foco na eficiência e na validação rigorosa do modelo.

### 2.1. Aquisição e Pré-processamento dos Dados

- **Dataset:** O projeto utilizou um dataset de 6.4 milhões de transações, das quais 500 mil foram amostradas de forma estratificada para o treinamento e a validação do modelo. Essa abordagem foi escolhida para reduzir o tempo computacional e garantir que a amostra representasse a distribuição de classes do conjunto de dados completo.
- **Engenharia de Features:** Uma única feature, `hour`, foi criada a partir da feature `step` para capturar a hora do dia da transação, uma variável potencialmente relevante para a detecção de fraude.

### 2.2. Treinamento e Seleção do Modelo

- **Modelos Candidatos:** Foram treinados e avaliados seis modelos de machine learning: Regressão Logística, Árvore de Decisão, Random Forest, AdaBoost, LightGBM e XGBoost.
- **Avaliação de Desempenho:** O desempenho dos modelos foi medido com validação cruzada, usando métricas como a área sob a curva ROC (AUC) e, mais importante, a área sob a curva Precision-Recall (AUC-PR), que é a métrica ideal para problemas com classes desbalanceadas.
- **Modelo Vencedor:** O **XGBoost** foi selecionado como o modelo de melhor desempenho devido à sua alta AUC-PR e sua capacidade de lidar com dados não lineares e desbalanceados.

## 3. Desafios e Decisões

### 3.1. Alto Desbalanceamento de Classes

O dataset apresentava um número de transações legítimas drasticamente maior do que o de transações fraudulentas. Para lidar com isso, foram adotadas as seguintes estratégias:
- Uso de uma **amostra estratificada** para garantir a presença de transações fraudulentas no conjunto de treino.
- Foco na métrica **AUC-PR** em vez da AUC-ROC, pois a AUC-PR é menos suscetível a resultados inflacionados em datasets desbalanceados.

### 3.2. Risco de Overfitting ou Vazamento de Dados

A alta performance do modelo XGBoost (AUC de 0.99) levantou a questão de possível overfitting ou vazamento de dados. Para mitigar essa preocupação:
- O modelo foi validado em um **conjunto de dados de teste completo (6.4 milhões de transações)**, o que demonstrou a sua capacidade de generalização e afastou a hipótese de um desempenho artificialmente alto.

## 4. Próximos Passos

Para aprimorar ainda mais o projeto, os seguintes passos são recomendados:
- **Otimização de Hiperparâmetros:** Realizar uma busca mais completa pelos melhores hiperparâmetros do XGBoost.
- **Validação Temporal:** Validar o modelo em conjuntos de dados com uma ordem temporal estrita para verificar sua adaptabilidade a novos padrões de fraude ao longo do tempo.